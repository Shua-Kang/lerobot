"""Controller tests that do not need a robot attached."""

import numpy as np
import pytest

from lerobot.scripts.lerobot_so101_gui import (
    OverheatError,
    ARM_JOINTS,
    MOTOR_NAMES,
    READY_POSE,
    RESET_POSE,
    ArmController,
    SerialPort,
    discover_so101_ports,
)
from lerobot.scripts.so101_kinematics import EEPose, UnreachableError


class FakeRobot:
    """A perfectly obedient arm: every command is reached immediately."""

    is_connected = True

    def __init__(self, pose):
        self.pose = dict(pose)
        self.actions = []

    def get_observation(self):
        return {f"{name}.pos": value for name, value in self.pose.items()}

    def send_action(self, action):
        self.actions.append(dict(action))
        for key, value in action.items():
            self.pose[key.removesuffix(".pos")] = value
        return action


@pytest.fixture
def controller(monkeypatch):
    pytest.importorskip("placo")
    monkeypatch.setattr("lerobot.scripts.lerobot_so101_gui.time.sleep", lambda _: None)
    arm = ArmController()
    arm.robot = FakeRobot(READY_POSE)
    arm._resync_target()
    return arm


def test_named_poses_cover_every_motor():
    for pose in (RESET_POSE, READY_POSE):
        assert set(pose) == set(MOTOR_NAMES)


def test_nudge_moves_one_axis_and_holds_the_rest(controller):
    before = controller.get_state()["ee"]
    after = controller.nudge("z", 0.02)["ee"]
    assert after["z"] == pytest.approx(before["z"] + 0.02, abs=1e-6)
    for axis in ("x", "y", "pitch", "roll"):
        assert after[axis] == pytest.approx(before[axis], abs=1e-6)


def test_nudge_rejects_an_unknown_axis(controller):
    with pytest.raises(ValueError, match="Unknown Cartesian axis"):
        controller.nudge("yaw", 1.0)


def test_repeated_nudges_track_the_commanded_target_exactly(controller):
    start = controller.get_state()["ee"]
    for _ in range(10):
        state = controller.nudge("x", 0.004)
    assert state["ee"]["x"] == pytest.approx(start["x"] + 0.04, abs=1e-6)
    assert state["ee"] == pytest.approx(state["target"], abs=1e-6)


def test_an_unreachable_request_sends_nothing(controller):
    controller.robot.actions.clear()
    with pytest.raises(UnreachableError):
        controller.goto_pose(EEPose(x=1.2, y=0.0, z=0.2, pitch=0.0, roll=0.0))
    assert controller.robot.actions == []


def test_a_refused_request_leaves_the_target_untouched(controller):
    target = controller.get_state()["target"]
    with pytest.raises(UnreachableError):
        controller.nudge("x", 1.5)
    assert controller.get_state()["target"] == pytest.approx(target, abs=1e-9)


def test_joint_nudge_is_clamped_to_the_joint_limit(controller):
    low, high = controller.kinematics().joint_limits["shoulder_pan"]
    state = controller.nudge_joint("shoulder_pan", 1000.0)
    assert state["joints"]["shoulder_pan"] == pytest.approx(high)
    state = controller.nudge_joint("shoulder_pan", -10_000.0)
    assert state["joints"]["shoulder_pan"] == pytest.approx(low)


def test_joint_nudge_resyncs_the_cartesian_target(controller):
    state = controller.nudge_joint("elbow_flex", -4.0)
    assert state["ee"] == pytest.approx(state["target"], abs=1e-9)


def test_every_command_holds_the_gripper_where_it_was(controller):
    controller.robot.pose["gripper"] = 42.0
    controller.nudge("z", 0.01)
    assert controller.robot.actions[-1]["gripper.pos"] == pytest.approx(42.0)


def test_set_gripper_reaches_the_endpoint_without_moving_the_arm(controller):
    joints_before = controller.get_state()["joints"]
    state = controller.set_gripper(100.0)
    assert state["joints"]["gripper"] == pytest.approx(100.0)
    assert len(controller.robot.actions) > 1
    for name in ARM_JOINTS:
        assert state["joints"][name] == pytest.approx(joints_before[name], abs=1e-9)


def test_set_gripper_clamps_to_the_normalised_range(controller):
    assert controller.set_gripper(500.0)["joints"]["gripper"] == pytest.approx(100.0)
    assert controller.set_gripper(-500.0)["joints"]["gripper"] == pytest.approx(0.0)


def test_named_pose_moves_end_on_the_exact_pose(controller):
    state = controller.goto_joint_pose(RESET_POSE, duration_s=0.05)
    assert state["joints"] == pytest.approx(RESET_POSE)
    assert state["ee"] == pytest.approx(state["target"], abs=1e-9)


def test_commands_need_a_connected_robot():
    arm = ArmController()
    with pytest.raises(RuntimeError, match="请先连接机械臂"):
        arm.nudge("z", 0.01)


def test_port_discovery_prefers_callout_devices(monkeypatch):
    class Port:
        def __init__(self, device, description):
            self.device, self.description, self.product = device, description, None

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_so101_gui.list_ports.comports",
        lambda: [Port("/dev/tty.usbmodem123", "SO-ARM"), Port("/dev/cu.Bluetooth-Incoming-Port", "BT")],
    )
    monkeypatch.setattr("lerobot.scripts.lerobot_so101_gui.Path.exists", lambda _: True)
    assert discover_so101_ports() == [SerialPort("/dev/cu.usbmodem123", "/dev/cu.usbmodem123  —  SO-ARM")]


def test_settle_gives_up_on_a_joint_that_cannot_move(controller, monkeypatch):
    """A blocked joint must not make the controller push forever."""
    stuck = np.array([float(READY_POSE[name]) for name in MOTOR_NAMES])
    monkeypatch.setattr(ArmController, "measured_joints", lambda self: stuck.copy())
    controller._settle(controller.robot, stuck[:5] + 30.0, 0.0)
    biases = [
        max(abs(action[f"{name}.pos"] - (stuck[i] + 30.0)) for i, name in enumerate(ARM_JOINTS))
        for action in controller.robot.actions
    ]
    assert biases and max(biases) <= 8.0


def test_settle_relaxes_a_stalled_joint_before_giving_up(controller, monkeypatch):
    """A joint that cannot reach its goal must not be left commanded past it.

    Holding an unreachable command is a stall, and a stalled Feetech servo heats
    until its overheat protection drops it off the bus.
    """
    stuck = np.array([float(READY_POSE[name]) for name in MOTOR_NAMES])
    monkeypatch.setattr(ArmController, "measured_joints", lambda self: stuck.copy())
    goal = stuck[:5] + 30.0
    controller.robot.actions.clear()
    controller._settle(controller.robot, goal, 0.0)
    final = controller.robot.actions[-1]
    assert [final[f"{name}.pos"] for name in ARM_JOINTS] == pytest.approx(list(goal))


def test_one_corrupt_temperature_packet_does_not_stop_the_arm(controller, monkeypatch):
    """Drive noise garbles the odd status byte; that must not look like an overheat."""
    readings = iter([500, 108, 39])  # out of band, then a one-off spike, then truth

    def fake_read(_data_name, _motor, **_kwargs):
        return next(readings, 39)

    controller.robot.bus = type("B", (), {"read": staticmethod(fake_read)})()
    monkeypatch.setattr("lerobot.scripts.lerobot_so101_gui.MOTOR_NAMES", ("shoulder_pan",))
    assert controller.check_temperatures() == {}


def test_a_confirmed_overheat_does_stop_the_arm(controller, monkeypatch):
    controller.robot.bus = type("B", (), {"read": staticmethod(lambda *a, **k: 71)})()
    monkeypatch.setattr("lerobot.scripts.lerobot_so101_gui.MOTOR_NAMES", ("shoulder_pan",))
    with pytest.raises(OverheatError, match="71"):
        controller.check_temperatures()
