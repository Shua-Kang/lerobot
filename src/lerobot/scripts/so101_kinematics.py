#!/usr/bin/env python

"""Exact analytic kinematics for the SO-ARM101 follower.

The SO-101 is a 5-DoF arm, so a generic 6-DoF Cartesian IK is over-constrained:
a least-squares solver spreads the unreachable part of the request across every
axis, which is why a "move +Z" request also drifts X, Y and the orientation.

The real structure is much friendlier than that:

* ``shoulder_pan`` rotates the whole arm about a vertical axis,
* ``shoulder_lift``/``elbow_flex``/``wrist_flex`` are three *parallel* joints
  forming a planar 3R chain inside the plane selected by the pan angle,
* ``wrist_roll`` spins the gripper about its own approach axis.

So the reachable task space is exactly ``(x, y, z, pitch, roll)``: yaw is not a
free variable, it is whatever the pan angle makes it.  Those five numbers map to
the five joints through a closed-form solution, which this module implements.
Every axis is therefore exactly decoupled -- asking for +5 mm of Z changes Z and
nothing else.

Link geometry is measured once from the official URDF (through placo's forward
kinematics) instead of being hard-coded, so the model can never drift away from
the URDF that the rest of LeRobot uses.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

import numpy as np

# Joint order used everywhere in this module and by the SO-101 follower.
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINTS = JOINT_NAMES[:-1]

# URDF frames, in chain order, that carry the joint axes we need.
_PAN_FRAME = "shoulder_link"
_LIFT_FRAME = "upper_arm_link"
_ELBOW_FRAME = "lower_arm_link"
_FLEX_FRAME = "wrist_link"
_ROLL_FRAME = "gripper_link"
_TIP_FRAME = "gripper_frame_link"


class UnreachableError(ValueError):
    """Raised when a requested pose is outside the arm's reachable set."""


@dataclass(frozen=True)
class EEPose:
    """End-effector pose in the five coordinates the SO-101 can actually track."""

    x: float
    y: float
    z: float
    pitch: float  # degrees, 0 = gripper horizontal, -90 = pointing straight down
    roll: float  # degrees, rotation of the gripper about its own approach axis

    def replace(self, **changes) -> EEPose:
        values = {"x": self.x, "y": self.y, "z": self.z, "pitch": self.pitch, "roll": self.roll}
        values.update(changes)
        return EEPose(**values)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "pitch": self.pitch, "roll": self.roll}


def _rot_y(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rot_z(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _unit(angle: float) -> np.ndarray:
    """In-plane unit vector, as (x, z), for a planar angle measured from +X to +Z."""
    return np.array((math.cos(angle), math.sin(angle)))


class SO101Kinematics:
    """Closed-form forward/inverse kinematics for one SO-101 follower arm."""

    def __init__(self, urdf_path: str):
        from lerobot.model.kinematics import RobotKinematics

        self._placo = RobotKinematics(urdf_path, joint_names=list(JOINT_NAMES))
        self._placo.forward_kinematics(np.zeros(len(JOINT_NAMES)))

        def frame(name: str) -> np.ndarray:
            return np.asarray(self._placo.robot.get_T_world_frame(name), dtype=float)

        pan, lift, elbow, flex = (frame(n) for n in (_PAN_FRAME, _LIFT_FRAME, _ELBOW_FRAME, _FLEX_FRAME))
        roll, tip = frame(_ROLL_FRAME), frame(_TIP_FRAME)

        # Vertical pan axis: every downstream point orbits this line.
        self.pan_center = np.array((pan[0, 3], pan[1, 3], 0.0))
        # A positive shoulder_pan command turns the arm about -Z in the URDF.
        self._pan_sign = -1.0 if pan[2, 2] < 0 else 1.0

        # Planar 3R chain, measured in the pan frame's XZ plane at zero pose.
        def xz(transform: np.ndarray) -> np.ndarray:
            return np.array((transform[0, 3], transform[2, 3]))

        self._shoulder_xz = xz(lift)
        link_1, link_2, link_3 = xz(elbow) - xz(lift), xz(flex) - xz(elbow), xz(roll) - xz(flex)
        self.l1, self.l2, self.l3 = (float(np.linalg.norm(v)) for v in (link_1, link_2, link_3))
        self._a1, self._a2, self._a3 = (math.atan2(v[1], v[0]) for v in (link_1, link_2, link_3))
        # The wrist_roll frame sits slightly off the arm plane; that offset is
        # constant, so it just shifts the plane the tip must be solved in.
        self._roll_axis_y = float(roll[1, 3])

        # Gripper tip relative to the wrist_roll frame, expressed in that frame.
        self._roll_rotation_0 = roll[:3, :3].copy()
        self._tip_offset = self._roll_rotation_0.T @ (tip[:3, 3] - roll[:3, 3])
        self._roll_to_tip_rotation = self._roll_rotation_0.T @ tip[:3, :3]

        self.reach_min = abs(self.l1 - self.l2)
        self.reach_max = self.l1 + self.l2
        self.joint_limits = _read_joint_limits(urdf_path)

    # ------------------------------------------------------------------ FK --

    def forward(self, joints_deg) -> EEPose:
        """Gripper pose for the five arm joints (degrees, JOINT_NAMES order)."""
        q = np.asarray(joints_deg, dtype=float)
        pan, lift, elbow, flex, roll = (math.radians(float(v)) for v in q[:5])
        tilt = lift + elbow + flex

        psi_1 = self._a1 - lift
        psi_2 = self._a2 - lift - elbow
        psi_3 = self._a3 - tilt
        roll_xz = self._shoulder_xz + self.l1 * _unit(psi_1) + self.l2 * _unit(psi_2) + self.l3 * _unit(psi_3)

        roll_rotation = _rot_y(tilt) @ self._roll_rotation_0 @ _rot_z(roll)
        tip_in_pan = np.array((roll_xz[0], self._roll_axis_y, roll_xz[1])) + roll_rotation @ self._tip_offset
        tip = self.pan_center + _rot_z(self._pan_sign * pan) @ (tip_in_pan - self.pan_center)
        return EEPose(
            x=float(tip[0]),
            y=float(tip[1]),
            z=float(tip[2]),
            pitch=math.degrees(psi_3),
            roll=math.degrees(roll),
        )

    def forward_matrix(self, joints_deg) -> np.ndarray:
        """4x4 gripper transform, matching ``RobotKinematics.forward_kinematics``."""
        q = np.asarray(joints_deg, dtype=float)
        pan, lift, elbow, flex, roll = (math.radians(float(v)) for v in q[:5])
        tilt = lift + elbow + flex
        pose = self.forward(q)
        rotation = (
            _rot_z(self._pan_sign * pan)
            @ _rot_y(tilt)
            @ self._roll_rotation_0
            @ _rot_z(roll)
            @ self._roll_to_tip_rotation
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = (pose.x, pose.y, pose.z)
        return transform

    # ------------------------------------------------------------------ IK --

    def inverse(self, pose: EEPose, current_joints_deg=None) -> np.ndarray:
        """Five arm joints (degrees) that put the gripper exactly at ``pose``.

        Raises ``UnreachableError`` when no exact solution exists, so the caller
        can reject the request instead of silently moving somewhere else.
        """
        return _nearest(self.inverse_branches(pose), current_joints_deg)

    def inverse_branches(self, pose: EEPose) -> list[np.ndarray]:
        """Both elbow-up / elbow-down solutions, in no particular order.

        On an SO-101 the second branch almost always folds the elbow past its
        stop, but solving for both keeps ``solve`` honest near the workspace
        edges, where the usable branch can be either one.
        """
        roll = math.radians(pose.roll)
        psi_3 = math.radians(pose.pitch)
        tilt = self._a3 - psi_3

        # Pitch and roll alone fix the wrist orientation, hence the constant
        # offset between the wrist_roll frame and the gripper tip.
        roll_rotation = _rot_y(tilt) @ self._roll_rotation_0 @ _rot_z(roll)
        tip_offset_in_pan = roll_rotation @ self._tip_offset

        # 1. Pan: the only unknown that can put the tip on the right side plane.
        target_y_in_pan = self._roll_axis_y + float(tip_offset_in_pan[1])
        sin_coefficient = -(pose.x - self.pan_center[0]) * self._pan_sign
        cos_coefficient = pose.y
        radius = math.hypot(sin_coefficient, cos_coefficient)
        if radius < 1e-9 or abs(target_y_in_pan) > radius:
            raise UnreachableError("目标点太靠近底座旋转轴，无法解算。")
        phase = math.atan2(cos_coefficient, sin_coefficient)
        base = math.asin(max(-1.0, min(1.0, target_y_in_pan / radius)))
        pan = min(
            (base - phase, math.pi - base - phase),
            key=lambda candidate: abs(_wrap(candidate)),
        )
        pan = _wrap(pan)

        # 2. Fold the target back into the arm plane and strip the wrist offset.
        tip_in_pan = self.pan_center + _rot_z(-self._pan_sign * pan) @ (
            np.array((pose.x, pose.y, pose.z)) - self.pan_center
        )
        roll_xz = np.array((tip_in_pan[0] - tip_offset_in_pan[0], tip_in_pan[2] - tip_offset_in_pan[2]))

        # 3. Planar 2R sub-problem for the wrist_flex joint centre.
        wrist_xz = roll_xz - self.l3 * _unit(psi_3)
        span = wrist_xz - self._shoulder_xz
        distance = float(np.linalg.norm(span))
        if distance > self.reach_max - 1e-9 or distance < self.reach_min + 1e-9:
            raise UnreachableError(
                f"目标超出可达范围（需要 {distance * 1000:.0f} mm，"
                f"可达 {self.reach_min * 1000:.0f}–{self.reach_max * 1000:.0f} mm）。"
            )
        direction = math.atan2(span[1], span[0])
        shoulder_opening = math.acos(
            max(-1.0, min(1.0, (distance**2 + self.l1**2 - self.l2**2) / (2 * distance * self.l1)))
        )
        elbow_opening = math.acos(
            max(-1.0, min(1.0, (self.l1**2 + self.l2**2 - distance**2) / (2 * self.l1 * self.l2)))
        )

        solutions = []
        for branch in (1.0, -1.0):
            psi_1 = direction + branch * shoulder_opening
            psi_2 = psi_1 - branch * (math.pi - elbow_opening)
            lift = self._a1 - psi_1
            elbow = self._a2 - psi_2 - lift
            flex = tilt - lift - elbow
            solutions.append(np.degrees([pan, lift, elbow, flex, roll]))
        return solutions

    # ------------------------------------------------------------- limits --

    def set_joint_limits(self, limits: dict[str, tuple[float, float]]) -> None:
        """Override the URDF ranges, e.g. with the arm's own calibrated travel."""
        self.joint_limits = {**self.joint_limits, **limits}

    def limit_violations(self, joints_deg) -> list[str]:
        """Names of arm joints whose commanded angle is outside the URDF range."""
        out = []
        for index, name in enumerate(ARM_JOINTS):
            low, high = self.joint_limits[name]
            value = float(joints_deg[index])
            if value < low - 1e-6 or value > high + 1e-6:
                out.append(f"{name} {value:.1f}° (允许 {low:.0f}°…{high:.0f}°)")
        return out

    def solve(self, pose: EEPose, current_joints_deg=None) -> np.ndarray:
        """``inverse``, restricted to branches the joints can actually reach."""
        branches = self.inverse_branches(pose)
        reachable = [candidate for candidate in branches if not self.limit_violations(candidate)]
        if reachable:
            return _nearest(reachable, current_joints_deg)
        nearest = _nearest(branches, current_joints_deg)
        raise UnreachableError("该姿态超出关节限位：" + "，".join(self.limit_violations(nearest)))


def _nearest(solutions: list[np.ndarray], current_joints_deg=None) -> np.ndarray:
    if current_joints_deg is None:
        return solutions[0]
    reference = np.asarray(current_joints_deg, dtype=float)[:5]
    return min(solutions, key=lambda candidate: float(np.linalg.norm(candidate - reference)))


def _read_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    root = ElementTree.parse(urdf_path).getroot()
    for joint in root.iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name in JOINT_NAMES and limit is not None:
            lower, upper = float(limit.get("lower", 0.0)), float(limit.get("upper", 0.0))
            limits[name] = (math.degrees(lower), math.degrees(upper))
    return limits


def _wrap(radians: float) -> float:
    return (radians + math.pi) % (2 * math.pi) - math.pi
