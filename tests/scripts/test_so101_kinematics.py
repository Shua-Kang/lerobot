"""Kinematics tests for the SO-101 desktop controller.

The whole point of the analytic solver is that one Cartesian request moves one
Cartesian coordinate, so that is what these tests pin down -- plus agreement
with the URDF model that the rest of LeRobot uses.
"""

import numpy as np
import pytest

from lerobot.scripts.lerobot_so101_gui import ensure_so101_urdf
from lerobot.scripts.so101_kinematics import (
    ARM_JOINTS,
    EEPose,
    SO101Kinematics,
    UnreachableError,
)

AXES = ("x", "y", "z", "pitch", "roll")


@pytest.fixture(scope="module")
def kinematics() -> SO101Kinematics:
    placo = pytest.importorskip("placo")  # noqa: F841
    return SO101Kinematics(str(ensure_so101_urdf()))


@pytest.fixture(scope="module")
def samples() -> np.ndarray:
    rng = np.random.default_rng(20240613)
    return np.column_stack(
        [
            rng.uniform(-90, 90, 300),
            rng.uniform(-90, 30, 300),
            rng.uniform(-90, 90, 300),
            rng.uniform(-80, 80, 300),
            rng.uniform(-150, 150, 300),
        ]
    )


def test_forward_matches_the_urdf_model(kinematics, samples):
    """The closed-form chain must agree with placo's URDF forward kinematics."""
    for joints in samples[:80]:
        expected = kinematics._placo.forward_kinematics(np.append(joints, 0.0))
        actual = kinematics.forward_matrix(joints)
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=2e-5)
        np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=1e-4)


def test_inverse_reproduces_the_pose_it_was_given(kinematics, samples):
    for joints in samples:
        pose = kinematics.forward(joints)
        try:
            solution = kinematics.inverse(pose, joints)
        except UnreachableError:
            continue
        np.testing.assert_allclose(
            [getattr(kinematics.forward(solution), axis) for axis in AXES],
            [getattr(pose, axis) for axis in AXES],
            atol=1e-6,
        )


@pytest.mark.parametrize(
    ("axis", "step"), [("x", 0.02), ("y", 0.02), ("z", 0.02), ("pitch", 10.0), ("roll", 25.0)]
)
def test_one_request_moves_exactly_one_coordinate(kinematics, axis, step):
    """The regression this solver exists for: no cross-axis bleed, at all."""
    start = np.array([10.0, -40.0, 70.0, 25.0, -15.0])
    before = kinematics.forward(start)
    after = kinematics.forward(
        kinematics.inverse(before.replace(**{axis: getattr(before, axis) + step}), start)
    )
    for other in AXES:
        expected = getattr(before, other) + (step if other == axis else 0.0)
        assert getattr(after, other) == pytest.approx(expected, abs=1e-6)


def test_repeated_steps_do_not_drift(kinematics):
    start = np.array([0.0, -35.0, 65.0, 30.0, 0.0])
    pose = kinematics.forward(start)
    joints, origin = start, pose
    for _ in range(20):
        pose = pose.replace(z=pose.z + 0.005)
        joints = kinematics.inverse(pose, joints)
    reached = kinematics.forward(joints)
    assert reached.z == pytest.approx(origin.z + 0.100, abs=1e-6)
    assert reached.x == pytest.approx(origin.x, abs=1e-6)
    assert reached.y == pytest.approx(origin.y, abs=1e-6)
    assert reached.pitch == pytest.approx(origin.pitch, abs=1e-6)


def test_out_of_reach_is_refused_rather_than_approximated(kinematics):
    with pytest.raises(UnreachableError):
        kinematics.inverse(EEPose(x=0.9, y=0.0, z=0.2, pitch=-45.0, roll=0.0))


def test_joint_limits_are_enforced_by_solve(kinematics):
    # Straight up at full stretch: geometrically fine, but folds a joint past
    # its stop, so `solve` must reject it while `inverse` still returns maths.
    pose = EEPose(x=0.05, y=0.0, z=0.36, pitch=85.0, roll=0.0)
    try:
        solution = kinematics.inverse(pose)
    except UnreachableError:
        pytest.skip("pose is out of reach, not a limit case")
    if not kinematics.limit_violations(solution):
        pytest.skip("pose happens to be within limits")
    with pytest.raises(UnreachableError, match="关节限位"):
        kinematics.solve(pose)


def test_both_elbow_branches_reach_the_pose(kinematics):
    pose = kinematics.forward(np.array([0.0, -35.0, 65.0, 30.0, 0.0]))
    branches = kinematics.inverse_branches(pose)
    assert len(branches) == 2
    assert not np.allclose(branches[0], branches[1])
    for solution in branches:
        reached = kinematics.forward(solution)
        assert (reached.x, reached.z) == pytest.approx((pose.x, pose.z), abs=1e-6)


def test_solve_only_returns_a_branch_the_joints_can_reach(kinematics):
    """The elbow-down branch folds past the stop, so solve must not hand it back."""
    pose = kinematics.forward(np.array([0.0, -35.0, 65.0, 30.0, 0.0]))
    far_reference = np.array([0.0, 130.0, -210.0, 145.0, 0.0])
    assert kinematics.limit_violations(kinematics.inverse(pose, far_reference))
    assert not kinematics.limit_violations(kinematics.solve(pose, far_reference))


def test_limits_cover_every_arm_joint(kinematics):
    assert set(ARM_JOINTS) <= set(kinematics.joint_limits)
    for low, high in kinematics.joint_limits.values():
        assert low < high
