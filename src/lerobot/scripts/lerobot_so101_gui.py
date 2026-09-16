#!/usr/bin/env python

"""macOS desktop controller for one SO-ARM101 follower arm.

Cartesian motion goes through :mod:`lerobot.scripts.so101_kinematics`, a
closed-form solver written for the SO-101's real 5-DoF structure, so each button
moves exactly one coordinate and leaves the other four untouched.

All motor I/O runs on a worker thread.  Importing this module never opens a
serial port, which keeps the port discovery and control math testable without a
robot attached.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
import traceback
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from serial.tools import list_ports

from lerobot.scripts.so101_kinematics import (
    ARM_JOINTS,
    JOINT_NAMES as MOTOR_NAMES,
    EEPose,
    SO101Kinematics,
    UnreachableError,
)

if TYPE_CHECKING:
    from lerobot.robots.so_follower import SO101Follower

ROBOT_ID = "so101_gui_follower"
# Official LeRobot SO-101 comfortable reset pose. Arm joints are degrees;
# gripper uses RANGE_0_100 (0 = closed, 100 = open).
RESET_POSE = {
    "shoulder_pan": -4.0,
    "shoulder_lift": -103.0,
    "elbow_flex": 97.0,
    "wrist_flex": 78.0,
    "wrist_roll": -65.0,
    "gripper": 0.0,
}
# A folded-but-upright pose that is a safe starting point for Cartesian jogging:
# well inside every joint limit and far from the 2R chain's singular extremes.
READY_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -35.0,
    "elbow_flex": 65.0,
    "wrist_flex": 30.0,
    "wrist_roll": 0.0,
    "gripper": 10.0,
}
# Joint-space slew used by every commanded motion, in degrees per 20 ms step.
DEGREES_PER_STEP = 1.2
STEP_PERIOD_S = 0.02
# LeRobot runs these Feetech servos at P=16, I=0, so gravity leaves a standing
# position error of a few degrees -- which is most of the residual Cartesian
# drift a user sees.  These bound the software integral that cancels it.
SETTLE_TOLERANCE_DEG = 0.3
SETTLE_ATTEMPTS = 26
SETTLE_GAIN = 0.6
SETTLE_MAX_BIAS_DEG = 8.0
SETTLE_PERIOD_S = 0.05
# A joint that moved less than this between two reads counts as standing still.
STILL_TOLERANCE_DEG = 0.15

CARTESIAN_AXES = ("x", "y", "z", "pitch", "roll")

# Feetech STS3215 servos trip their overheat protection around 70 C and drop off
# the bus mid-packet, which surfaces as a confusing serial read failure. Warn
# well before that, and refuse to keep driving a joint that is nearly there.
TEMPERATURE_WARN_C = 50
TEMPERATURE_STOP_C = 62


class OverheatError(RuntimeError):
    """A servo is too hot to keep driving."""


@dataclass(frozen=True)
class SerialPort:
    device: str
    label: str


def discover_so101_ports() -> list[SerialPort]:
    """Return likely USB serial devices, preferring the call-out device on macOS."""
    found: dict[str, SerialPort] = {}
    for port in list_ports.comports():
        device = port.device
        lowered = device.lower()
        if not any(token in lowered for token in ("usbmodem", "usbserial", "ttyacm", "ttyusb")):
            continue
        # macOS exposes the same device as /dev/tty.* and /dev/cu.*.  pyserial
        # recommends /dev/cu.* for initiating an outbound connection.
        key = device.replace("/dev/tty.", "/dev/cu.")
        preferred = key if Path(key).exists() else device
        detail = port.description or port.product or "USB serial"
        found[preferred] = SerialPort(preferred, f"{preferred}  —  {detail}")
    return sorted(found.values(), key=lambda item: item.device)


def calibration_file(robot_id: str = ROBOT_ID) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / "calibration" / "robots" / "so_follower" / f"{robot_id}.json"


def ensure_so101_urdf() -> Path:
    """Fetch the official SO-101 URDF bundle once and cache it."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    destination = HF_LEROBOT_HOME / "robot-urdfs" / "so101"
    urdf = destination / "so101_new_calib.urdf"
    marker = destination / ".sync_complete"
    if not marker.exists():
        archive = destination.parent / "so-arm100-main.zip"
        destination.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/TheRobotStudio/SO-ARM100/archive/refs/heads/main.zip", archive
        )
        prefix = "SO-ARM100-main/Simulation/SO101/"
        try:
            with zipfile.ZipFile(archive) as bundle:
                members = [
                    name for name in bundle.namelist() if name.startswith(prefix) and not name.endswith("/")
                ]
                if not members:
                    raise FileNotFoundError("官方 SO101 压缩包中没有找到 Simulation/SO101")
                for member in members:
                    relative = Path(member.removeprefix(prefix))
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        finally:
            archive.unlink(missing_ok=True)
        marker.touch()
    if not urdf.is_file():
        raise FileNotFoundError(f"SO-101 URDF download is incomplete: {urdf}")
    return urdf


class ArmController:
    """Owns the serial connection. Methods must be called from one worker thread."""

    def __init__(self, robot_id: str = ROBOT_ID):
        self.robot_id = robot_id
        self.robot: SO101Follower | None = None
        self._calibration_robot: SO101Follower | None = None
        self._range_min: dict[str, int] = {}
        self._range_max: dict[str, int] = {}
        self._homings: dict[str, int] = {}
        self._kinematics: SO101Kinematics | None = None
        # The pose the user has asked for, kept separate from the measured pose
        # so that repeated 5 mm steps stay exact instead of accumulating the
        # servos' following error.
        self._target: EEPose | None = None

    @staticmethod
    def _make_robot(port: str, robot_id: str, max_relative_target: float | None = None):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=True,
            max_relative_target=max_relative_target,
        )
        return SO101Follower(config)

    # --------------------------------------------------------- connection --

    def connect(self, port: str) -> dict:
        if not calibration_file(self.robot_id).is_file():
            raise RuntimeError("还没有校准文件，请先完成校准。")
        self.disconnect()
        robot = self._make_robot(port, self.robot_id, max_relative_target=12.0)
        try:
            robot.connect(calibrate=False)
            if not robot.is_calibrated:
                raise RuntimeError("电机中的校准值与校准文件不一致，请重新校准。")
        except Exception:
            if robot.is_connected:
                robot.disconnect()
            raise
        self.robot = robot
        self._apply_calibrated_limits()
        self._resync_target()
        return self.get_state()

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
            finally:
                self.robot = None
                self._target = None
        self.abort_calibration()

    # -------------------------------------------------------- calibration --

    def begin_calibration(self, port: str) -> None:
        self.disconnect()
        from lerobot.motors.feetech import OperatingMode

        robot = self._make_robot(port, self.robot_id)
        robot.bus.connect()
        robot.bus.disable_torque()
        for motor in robot.bus.motors:
            robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        self._calibration_robot = robot

    def set_calibration_center(self) -> dict[str, int]:
        robot = self._require_calibration()
        self._homings = {str(k): int(v) for k, v in robot.bus.set_half_turn_homings().items()}
        positions = robot.bus.sync_read("Present_Position", normalize=False)
        self._range_min = {str(k): int(v) for k, v in positions.items()}
        self._range_max = self._range_min.copy()
        return positions

    def sample_calibration(self) -> dict[str, int]:
        robot = self._require_calibration()
        positions = robot.bus.sync_read("Present_Position", normalize=False)
        for name, value in positions.items():
            if name == "wrist_roll":
                continue
            self._range_min[name] = min(self._range_min[name], int(value))
            self._range_max[name] = max(self._range_max[name], int(value))
        return {str(k): int(v) for k, v in positions.items()}

    def finish_calibration(self) -> Path:
        from lerobot.motors import MotorCalibration

        robot = self._require_calibration()
        if not self._homings:
            raise RuntimeError("请先设置中心点。")
        self._range_min["wrist_roll"], self._range_max["wrist_roll"] = 0, 4095
        calibration = {
            name: MotorCalibration(
                id=robot.bus.motors[name].id,
                drive_mode=0,
                homing_offset=self._homings[name],
                range_min=self._range_min[name],
                range_max=self._range_max[name],
            )
            for name in MOTOR_NAMES
        }
        too_small = [
            name
            for name in MOTOR_NAMES
            if name != "wrist_roll" and self._range_max[name] - self._range_min[name] < 50
        ]
        if too_small:
            raise RuntimeError("这些关节的活动范围过小，请重新扫过完整范围：" + ", ".join(too_small))
        robot.bus.write_calibration(calibration)
        robot.calibration = calibration
        robot._save_calibration()
        path = robot.calibration_fpath
        self.abort_calibration()
        return path

    def abort_calibration(self) -> None:
        robot, self._calibration_robot = self._calibration_robot, None
        if robot is not None and robot.bus.is_connected:
            robot.bus.disconnect(True)
        self._range_min, self._range_max, self._homings = {}, {}, {}

    def reset_motor_centers(self, port: str) -> None:
        """Clear stale offsets and center all motors on the current physical pose."""
        self.disconnect()
        robot = self._make_robot(port, self.robot_id)
        try:
            robot.bus.connect()
            robot.bus.disable_torque()
            robot.bus.set_half_turn_homings()
        finally:
            if robot.bus.is_connected:
                robot.bus.disconnect(True)
        path = calibration_file(self.robot_id)
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------ control --

    def kinematics(self) -> SO101Kinematics:
        if self._kinematics is None:
            self._kinematics = SO101Kinematics(str(ensure_so101_urdf()))
        return self._kinematics

    def _apply_calibrated_limits(self) -> None:
        """Trust this arm's own calibrated travel over the generic URDF ranges.

        ``MotorNormMode.DEGREES`` maps the calibrated tick range symmetrically
        around its midpoint, so the reachable angle span follows directly from
        the calibration the user swept -- and it is usually a little wider than
        the conservative limits shipped in the URDF.
        """
        robot = self._require_robot()
        limits = {}
        for name, calibration in (robot.calibration or {}).items():
            if name not in ARM_JOINTS:
                continue
            half_span = (calibration.range_max - calibration.range_min) * 360.0 / 4095.0 / 2.0
            limits[name] = (-half_span, half_span)
        if limits:
            self.kinematics().set_joint_limits(limits)

    def measured_joints(self) -> np.ndarray:
        observation = self._require_robot().get_observation()
        return np.array([float(observation[f"{name}.pos"]) for name in MOTOR_NAMES], dtype=float)

    def nudge(self, axis: str, amount: float) -> dict:
        """Move one Cartesian coordinate and hold the other four exactly."""
        if axis not in CARTESIAN_AXES:
            raise ValueError(f"Unknown Cartesian axis: {axis}")
        if self._target is None:
            self._resync_target()
        target = self._target.replace(**{axis: getattr(self._target, axis) + amount})
        return self.goto_pose(target)

    def goto_pose(self, pose: EEPose) -> dict:
        joints = self.measured_joints()
        # UnreachableError propagates to the UI: refusing a request the arm
        # cannot honour is the whole point, a least-squares "nearest" answer is
        # what makes the other axes drift.
        solution = self.kinematics().solve(pose, joints)
        self._glide(solution, float(joints[5]))
        self._target = pose
        return self.get_state()

    def nudge_joint(self, name: str, degrees: float) -> dict:
        """Direct single-joint jog, for recovery when Cartesian IK is blocked."""
        joints = self.measured_joints()
        index = MOTOR_NAMES.index(name)
        solution = joints[:5].copy()
        solution[index] += degrees
        low, high = self.kinematics().joint_limits[name]
        solution[index] = max(low, min(high, solution[index]))
        self._glide(solution, float(joints[5]))
        self._resync_target()
        return self.get_state()

    def _glide(self, arm_joints_deg: np.ndarray, gripper: float) -> None:
        """Ramp to a joint target at a bounded speed, then drive out the droop."""
        robot = self._require_robot()
        start = self.measured_joints()[:5]
        goal = np.asarray(arm_joints_deg, dtype=float)[:5]
        steps = max(1, int(np.max(np.abs(goal - start)) / DEGREES_PER_STEP) + 1)
        for step in range(1, steps + 1):
            waypoint = start + (goal - start) * (step / steps)
            self._send_arm(robot, waypoint, gripper)
            time.sleep(STEP_PERIOD_S)
        self._settle(robot, goal, gripper)

    def _settle(self, robot, goal: np.ndarray, gripper: float) -> None:
        """Integral outer loop: push the command past the goal until it is met.

        A P-only servo under load stops short of its goal and stays there, so
        the arm ends up a few millimetres away from an exactly solved pose.
        Biasing the command by the accumulated error removes that offset without
        touching the servo gains LeRobot deliberately picked.

        The bias only ever grows while the arm is standing still: correcting a
        joint that is still flying would chase its velocity rather than its
        steady-state error, and that is what turns a 10 mm request into 13 mm.
        """
        bias = np.zeros(5)
        previous = self.measured_joints()[:5]
        best, stagnant = np.inf, 0
        for _ in range(SETTLE_ATTEMPTS):
            time.sleep(SETTLE_PERIOD_S)
            joints = self.measured_joints()[:5]
            moving = float(np.max(np.abs(joints - previous))) >= STILL_TOLERANCE_DEG
            previous = joints
            if moving:
                continue
            error = goal - joints
            worst = float(np.max(np.abs(error)))
            if worst <= SETTLE_TOLERANCE_DEG:
                return
            if worst > best - 0.02:
                # Not converging: against a hard stop, or the load is beyond the
                # servo. Give up after a few tries instead of pushing harder,
                # and drop the bias first -- leaving a joint commanded past a
                # stop it cannot pass is a stall, and a stalled Feetech servo
                # heats until its overheat protection drops it off the bus.
                stagnant += 1
                if stagnant >= 3:
                    self._send_arm(robot, goal, gripper)
                    return
            else:
                stagnant = 0
            best = min(best, worst)
            # The error changes sign once a joint goes past the goal, so the same
            # accumulator that removes droop also unwinds an overshoot.
            bias = np.clip(bias + SETTLE_GAIN * error, -SETTLE_MAX_BIAS_DEG, SETTLE_MAX_BIAS_DEG)
            self._send_arm(robot, goal + bias, gripper)

    def _still_joints(self, timeout_s: float = 0.6) -> np.ndarray:
        """Read joints once the arm has stopped moving, so poses are not read mid-flight."""
        previous = self.measured_joints()
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            time.sleep(0.04)
            current = self.measured_joints()
            if float(np.max(np.abs(current - previous))) < STILL_TOLERANCE_DEG:
                return current
            previous = current
        return previous

    @staticmethod
    def _send_arm(robot, arm_joints_deg: np.ndarray, gripper: float) -> None:
        action = {f"{name}.pos": float(arm_joints_deg[index]) for index, name in enumerate(ARM_JOINTS)}
        action["gripper.pos"] = gripper
        robot.send_action(action)

    def set_gripper(self, value: float, duration_s: float = 0.8) -> dict:
        robot = self._require_robot()
        target = max(0.0, min(100.0, value))
        joints = self.measured_joints()
        current = float(joints[5])
        steps = max(1, int(abs(target - current) / 3.0) + 1)
        held = {f"{name}.pos": float(joints[index]) for index, name in enumerate(ARM_JOINTS)}
        for step in range(1, steps + 1):
            action = dict(held)
            action["gripper.pos"] = current + (target - current) * (step / steps)
            robot.send_action(action)
            time.sleep(duration_s / steps)
        return self.get_state()

    def adjust_gripper(self, delta: float) -> dict:
        return self.set_gripper(float(self.measured_joints()[5]) + delta)

    def goto_joint_pose(self, pose: dict[str, float], duration_s: float = 3.0) -> dict:
        """Slow joint-space move, used for the named home poses."""
        robot = self._require_robot()
        start = self.measured_joints()
        goal = np.array([pose[name] for name in MOTOR_NAMES], dtype=float)
        steps = max(1, int(duration_s / STEP_PERIOD_S))
        for step in range(1, steps + 1):
            waypoint = start + (goal - start) * (step / steps)
            robot.send_action({f"{name}.pos": float(waypoint[i]) for i, name in enumerate(MOTOR_NAMES)})
            time.sleep(duration_s / steps)
        time.sleep(0.2)
        self._resync_target()
        return self.get_state()

    def reset_pose(self) -> dict:
        return self.goto_joint_pose(RESET_POSE)

    def ready_pose(self) -> dict:
        return self.goto_joint_pose(READY_POSE)

    def _resync_target(self) -> None:
        self._target = self.kinematics().forward(self.measured_joints())

    def temperatures(self) -> dict[str, int]:
        robot = self._require_robot()
        try:
            return {
                name: int(robot.bus.read("Present_Temperature", name, normalize=False))
                for name in MOTOR_NAMES
            }
        except Exception:
            return {}

    def check_temperatures(self) -> dict[str, int]:
        """Raise before a servo cooks itself; return the joints that are warm."""
        temps = self.temperatures()
        hot = {name: value for name, value in temps.items() if value >= TEMPERATURE_STOP_C}
        if hot:
            raise OverheatError(
                "电机过热，已停止运动，请断电冷却几分钟："
                + "，".join(f"{name} {value}°C" for name, value in hot.items())
            )
        return {name: value for name, value in temps.items() if value >= TEMPERATURE_WARN_C}

    def get_state(self) -> dict:
        joints = self._still_joints()
        actual = self.kinematics().forward(joints)
        return {
            "joints": {name: float(joints[index]) for index, name in enumerate(MOTOR_NAMES)},
            "ee": actual.as_dict(),
            "target": self._target.as_dict() if self._target is not None else None,
            "warm": self.check_temperatures(),
        }

    def _require_robot(self) -> SO101Follower:
        if self.robot is None or not self.robot.is_connected:
            raise RuntimeError("请先连接机械臂。")
        return self.robot

    def _require_calibration(self) -> SO101Follower:
        if self._calibration_robot is None or not self._calibration_robot.bus.is_connected:
            raise RuntimeError("校准尚未开始。")
        return self._calibration_robot


class SO101App:
    POLL_MS = 700

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("SO-ARM101 控制器")
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"860x{min(900, screen_height - 100)}")
        self.root.minsize(720, 560)
        self.controller = ArmController()
        self.jobs: queue.Queue[tuple[Callable, Callable | None]] = queue.Queue()
        self.results: queue.Queue[tuple[bool, object, Callable | None]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.port_by_label: dict[str, str] = {}
        self.last_ports: tuple[str, ...] = ()
        self.calibration_stage = 0
        self.sampling = False
        self.connected_port: str | None = None
        self.closing = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._drain_results)
        self.root.after(0, self._poll_ports)

    # ---------------------------------------------------------------- UI --

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        page = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=page.yview)
        page.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        page.pack(side="left", fill="both", expand=True)
        outer = ttk.Frame(page, padding=18)
        page_window = page.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", lambda _: page.configure(scrollregion=page.bbox("all")))
        page.bind("<Configure>", lambda event: page.itemconfigure(page_window, width=event.width))

        def scroll(event) -> None:
            if event.delta:
                page.yview_scroll(-1 if event.delta > 0 else 1, "units")

        page.bind_all("<MouseWheel>", scroll)
        ttk.Label(outer, text="SO-ARM101 单臂控制", font=("Helvetica", 22, "bold")).pack(anchor="w")
        ttk.Label(outer, text="自动识别 USB · 图形化校准 · 解析式笛卡尔控制（单轴解耦）").pack(
            anchor="w", pady=(2, 14)
        )

        device = ttk.LabelFrame(outer, text="1. 设备", padding=12)
        device.pack(fill="x")
        self.port_var = tk.StringVar(value="正在扫描…")
        self.port_box = ttk.Combobox(device, textvariable=self.port_var, state="readonly", width=64)
        self.port_box.grid(row=0, column=0, sticky="ew")
        self.connect_btn = ttk.Button(device, text="连接", command=self._toggle_connection)
        self.connect_btn.grid(row=0, column=1, padx=(8, 0))
        device.columnconfigure(0, weight=1)
        self.device_status = ttk.Label(device, text="等待 USB 机械臂")
        self.device_status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        calibration = ttk.LabelFrame(outer, text="2. 校准", padding=12)
        calibration.pack(fill="x", pady=12)
        self.cal_status = ttk.Label(calibration, text=self._calibration_status_text())
        self.cal_status.pack(anchor="w")
        row = ttk.Frame(calibration)
        row.pack(fill="x", pady=(8, 0))
        self.cal_btn = ttk.Button(row, text="开始校准", command=self._calibration_next)
        self.cal_btn.pack(side="left")
        ttk.Button(row, text="重置电机中心点", command=self._confirm_reset).pack(side="left", padx=8)

        self.move_widgets: list[object] = []
        self._build_cartesian(outer)
        self._build_joints(outer)

        log_frame = ttk.LabelFrame(outer, text="状态", padding=8)
        log_frame.pack(fill="x", pady=(12, 0))
        self.log = tk.Text(log_frame, height=5, state="disabled", wrap="word")
        self.log.pack(fill="x")
        self._set_controls(False)

    def _build_cartesian(self, outer) -> None:
        tk, ttk = self.tk, self.ttk
        control = ttk.LabelFrame(outer, text="3. 末端笛卡尔控制", padding=12)
        control.pack(fill="x")

        steps = ttk.Frame(control)
        steps.pack(fill="x")
        self.step_mm = tk.DoubleVar(value=5.0)
        self.step_deg = tk.DoubleVar(value=5.0)
        ttk.Label(steps, text="平移步长").pack(side="left")
        for value in (1.0, 5.0, 10.0, 20.0):
            ttk.Radiobutton(steps, text=f"{value:g} mm", value=value, variable=self.step_mm).pack(
                side="left", padx=2
            )
        ttk.Label(steps, text="    旋转步长").pack(side="left", padx=(12, 0))
        for value in (1.0, 5.0, 10.0):
            ttk.Radiobutton(steps, text=f"{value:g}°", value=value, variable=self.step_deg).pack(
                side="left", padx=2
            )

        pad = ttk.Frame(control)
        pad.pack(pady=(10, 0))
        layout = (
            ("X− 后", "x", -1, 0, 0),
            ("X+ 前", "x", 1, 0, 1),
            ("Y− 右", "y", -1, 1, 0),
            ("Y+ 左", "y", 1, 1, 1),
            ("Z− 下", "z", -1, 2, 0),
            ("Z+ 上", "z", 1, 2, 1),
            ("Pitch− 低头", "pitch", -1, 0, 2),
            ("Pitch+ 抬头", "pitch", 1, 0, 3),
            ("Roll− 逆时针", "roll", -1, 1, 2),
            ("Roll+ 顺时针", "roll", 1, 1, 3),
        )
        for label, axis, sign, r, c in layout:
            button = ttk.Button(pad, text=label, width=14, command=lambda a=axis, s=sign: self._nudge(a, s))
            button.grid(row=r, column=c, padx=4, pady=4, ipadx=6, ipady=3)
            self.move_widgets.append(button)

        grip = ttk.Frame(control)
        grip.pack(pady=(12, 0))
        ttk.Label(grip, text="夹爪 ").pack(side="left")
        for label, action in (
            ("全关", lambda: self._submit_move(lambda: self.controller.set_gripper(0.0))),
            ("−5", lambda: self._submit_move(lambda: self.controller.adjust_gripper(-5.0))),
            ("+5", lambda: self._submit_move(lambda: self.controller.adjust_gripper(5.0))),
            ("全开", lambda: self._submit_move(lambda: self.controller.set_gripper(100.0))),
        ):
            button = ttk.Button(grip, text=label, width=7, command=action)
            button.pack(side="left", padx=3)
            self.move_widgets.append(button)

        homes = ttk.Frame(control)
        homes.pack(pady=(12, 0))
        for label, method_name, note in (
            ("回到 Ready Pose", "ready_pose", "一个适合笛卡尔微调的直立姿态"),
            ("回到 Reset Pose", "reset_pose", "LeRobot 标准收拢姿态"),
        ):
            button = ttk.Button(
                homes, text=label, command=lambda m=method_name, n=note, t=label: self._home(m, t, n)
            )
            button.pack(side="left", padx=6, ipadx=10, ipady=3)
            self.move_widgets.append(button)

        self.pose_var = tk.StringVar(value="未连接")
        ttk.Label(
            control, textvariable=self.pose_var, wraplength=760, font=("Menlo", 11), justify="left"
        ).pack(anchor="w", pady=(12, 0))

    def _build_joints(self, outer) -> None:
        ttk = self.ttk
        frame = ttk.LabelFrame(outer, text="4. 关节微调（笛卡尔解不出来时用它脱困）", padding=12)
        frame.pack(fill="x", pady=(12, 0))
        for index, name in enumerate(ARM_JOINTS):
            ttk.Label(frame, text=name, width=15).grid(row=index, column=0, sticky="w", pady=2)
            for column, degrees in enumerate((-5.0, -1.0, 1.0, 5.0)):
                button = ttk.Button(
                    frame,
                    text=f"{degrees:+g}°",
                    width=6,
                    command=lambda n=name, d=degrees: self._submit_move(
                        lambda: self.controller.nudge_joint(n, d)
                    ),
                )
                button.grid(row=index, column=column + 1, padx=3, pady=2)
                self.move_widgets.append(button)

    # ----------------------------------------------------------- actions --

    def _nudge(self, axis: str, sign: int) -> None:
        amount = sign * (self.step_mm.get() / 1000.0 if axis in "xyz" else self.step_deg.get())
        self._submit_move(lambda: self.controller.nudge(axis, amount))

    def _home(self, method_name: str, title: str, note: str) -> None:
        from tkinter import messagebox

        if not messagebox.askyesno(
            title, f"机械臂将缓慢移动到「{title}」（{note}）。\n\n请确认路径无障碍。是否继续？"
        ):
            return
        self._submit_move(lambda: getattr(self.controller, method_name)())

    def _submit_move(self, function: Callable) -> None:
        self._set_controls(False)
        self._submit(function, self._movement_done)

    def _selected_port(self) -> str:
        port = self.port_by_label.get(self.port_var.get())
        if not port:
            raise RuntimeError("没有检测到 SO-ARM101 USB 串口。")
        return port

    def _poll_ports(self) -> None:
        ports = discover_so101_ports()
        devices = tuple(item.device for item in ports)
        if devices != self.last_ports:
            self.last_ports = devices
            self.port_by_label = {item.label: item.device for item in ports}
            labels = list(self.port_by_label)
            self.port_box["values"] = labels
            current = self.connected_port
            if current and current not in devices:
                self.connected_port = None
                self._set_controls(False)
                self.connect_btn.config(text="连接")
                self.device_status.config(text="USB 已拔出，正在安全断开…")
                self._submit(self.controller.disconnect, lambda _: self._log("机械臂已断开"))
            if labels and self.port_var.get() not in labels:
                self.port_var.set(labels[0])
                self.device_status.config(text="已自动检测到机械臂串口")
            elif not labels:
                self.port_var.set("未检测到 USB 串口")
                self.device_status.config(text="请连接机械臂 USB 和正确的外部电源")
            self._log("USB 设备变化：" + (", ".join(devices) if devices else "无"))
        if not self.stop_event.is_set():
            self.root.after(self.POLL_MS, self._poll_ports)

    def _toggle_connection(self) -> None:
        if self.connected_port:
            self._submit(self.controller.disconnect, self._disconnected)
            return
        try:
            port = self._selected_port()
        except Exception as error:
            self._show_error(error)
            return
        self.device_status.config(text="正在连接…")
        self._submit(lambda: self.controller.connect(port), lambda state: self._connected(port, state))

    def _connected(self, port: str, state: object) -> None:
        self.connected_port = port
        self.connect_btn.config(text="断开")
        self.device_status.config(text=f"已连接：{port}")
        self._set_controls(True)
        self._show_state(state)

    def _disconnected(self, _: object) -> None:
        self.connected_port = None
        self.connect_btn.config(text="连接")
        self.device_status.config(text="已断开")
        self.pose_var.set("未连接")
        self._set_controls(False)

    def _calibration_next(self) -> None:
        try:
            port = self._selected_port()
        except Exception as error:
            self._show_error(error)
            return
        if self.calibration_stage == 0:
            self._submit(lambda: self.controller.begin_calibration(port), self._calibration_started)
        elif self.calibration_stage == 1:
            self._submit(self.controller.set_calibration_center, self._center_set)
        else:
            self.sampling = False
            self._submit(self.controller.finish_calibration, self._calibration_finished)

    def _calibration_started(self, _: object) -> None:
        self.calibration_stage = 1
        self.cal_btn.config(text="中心姿态已摆好")
        self.cal_status.config(text="扭矩已关闭。把每个关节移到活动范围的中间，然后点击按钮。")
        self._log("校准开始：请把机械臂摆到各关节的中间位置")

    def _center_set(self, _: object) -> None:
        self.calibration_stage = 2
        self.cal_btn.config(text="完成范围采集")
        self.cal_status.config(
            text="依次缓慢移动除 Wrist Roll 外的每个关节，扫过完整安全范围，然后点击完成。"
        )
        self.sampling = True
        self._sample_once()
        self._log("中心点已设置，正在采集关节活动范围")

    def _sample_once(self) -> None:
        if not self.sampling:
            return
        self._submit(self.controller.sample_calibration, lambda _: self.root.after(60, self._sample_once))

    def _calibration_finished(self, path: object) -> None:
        self.calibration_stage = 0
        self.cal_btn.config(text="重新校准")
        self.cal_status.config(text=f"校准完成：{path}")
        self._log(f"校准文件已保存：{path}")

    def _confirm_reset(self) -> None:
        from tkinter import messagebox

        if not messagebox.askyesno(
            "重置电机中心点",
            "此操作会关闭扭矩，把当前物理姿态设为所有电机的中心，并删除旧校准文件。\n\n"
            "请先扶住机械臂、摆到各关节的中间位置。完成后必须重新校准。是否继续？",
        ):
            return
        try:
            port = self._selected_port()
        except Exception as error:
            self._show_error(error)
            return
        self._submit(lambda: self.controller.reset_motor_centers(port), self._reset_finished)

    def _reset_finished(self, _: object) -> None:
        self.cal_status.config(text="中心点已重置；旧校准已删除。现在请点击“开始校准”。")
        self.cal_btn.config(text="开始校准")
        self.calibration_stage = 0
        self._log("电机中心点已重置")

    def _movement_done(self, state: object) -> None:
        self._set_controls(True)
        self._show_state(state)

    def _show_state(self, state: object) -> None:
        if not isinstance(state, dict) or "ee" not in state:
            return
        ee, target = state["ee"], state.get("target")
        lines = [
            "实际位置  "
            + "  ".join(f"{axis.upper()} {ee[axis] * 1000:8.1f} mm" for axis in "xyz")
            + f"   Pitch {ee['pitch']:7.1f}°   Roll {ee['roll']:7.1f}°"
        ]
        if target:
            lines.append(
                "指令位置  "
                + "  ".join(f"{axis.upper()} {target[axis] * 1000:8.1f} mm" for axis in "xyz")
                + f"   Pitch {target['pitch']:7.1f}°   Roll {target['roll']:7.1f}°"
            )
            lines.append(
                "跟随误差  "
                + "  ".join(f"{axis.upper()} {(ee[axis] - target[axis]) * 1000:+8.1f} mm" for axis in "xyz")
                + f"   Pitch {ee['pitch'] - target['pitch']:+7.1f}°"
                + f"   Roll {ee['roll'] - target['roll']:+7.1f}°"
            )
        lines.append(
            "关节角度  " + "  ".join(f"{name}={value:.1f}°" for name, value in state["joints"].items())
        )
        warm = state.get("warm")
        if warm:
            lines.append("⚠ 电机偏热  " + "  ".join(f"{n} {v}°C" for n, v in warm.items()))
        self.pose_var.set("\n".join(lines))

    def _set_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self.move_widgets:
            widget.config(state=state)

    def _calibration_status_text(self) -> str:
        path = calibration_file()
        return f"已找到校准文件：{path}" if path.is_file() else "未找到校准文件，需要先校准。"

    def _submit(self, function: Callable, callback: Callable | None = None) -> None:
        self.jobs.put((function, callback))

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                function, callback = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.results.put((True, function(), callback))
            except Exception as error:
                error.__notes__ = [traceback.format_exc()]
                self.results.put((False, error, callback))

    def _drain_results(self) -> None:
        try:
            while True:
                ok, result, callback = self.results.get_nowait()
                if ok:
                    if callback:
                        callback(result)
                else:
                    self.sampling = False
                    self._set_controls(bool(self.connected_port))
                    self._show_error(result)
                    if self.connected_port:
                        self._submit(self.controller.get_state, self._show_state)
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.root.after(80, self._drain_results)

    def _show_error(self, error: object) -> None:
        from tkinter import messagebox

        self._log(f"错误：{error}")
        if isinstance(error, OverheatError):
            title = "电机过热"
        elif isinstance(error, UnreachableError):
            title = "该动作超出机械臂能力"
        else:
            title = "操作失败"
        messagebox.showerror(title, str(error))

    def _log(self, message: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S ") + message + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.sampling = False
        self._set_controls(False)
        self.device_status.config(text="正在安全断开…")
        self._submit(self.controller.disconnect, self._finish_close)

    def _finish_close(self, _: object) -> None:
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    SO101App().run()


if __name__ == "__main__":
    main()
