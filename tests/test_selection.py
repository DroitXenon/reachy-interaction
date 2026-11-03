import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection import Detection  # noqa: E402
from pointing import Pointing, classify_pointing  # noqa: E402
from robot import image_point_to_head_angles  # noqa: E402
from selection import (  # noqa: E402
    STABLE_FRAMES,
    Selection,
    Status,
    TargetStabilizer,
    ray_hits_box,
    select_target,
)

DIAG = float(np.hypot(640, 480))


def det(label, x1, y1, x2, y2):
    return Detection(label, 0.9, np.array([x1, y1, x2, y2], dtype=float))


def pointing_right(origin=(100, 240)):
    return Pointing(np.array(origin, dtype=float), np.array([1.0, 0.0]))


def test_ray_hits_box_forward_only():
    o, d = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert ray_hits_box(o, d, np.array([10, -5, 20, 5]))
    assert not ray_hits_box(o, d, np.array([-20, -5, -10, 5]))  # behind
    assert not ray_hits_box(o, d, np.array([10, 10, 20, 20]))  # off to the side


def test_selects_object_on_ray():
    cup, bottle = det("cup", 400, 200, 460, 280), det("bottle", 400, 20, 440, 100)
    sel = select_target(pointing_right(), [cup, bottle], DIAG)
    assert sel.status is Status.SELECTED and sel.target is cup


def test_pointing_at_other_object():
    cup, bottle = det("cup", 400, 200, 460, 280), det("bottle", 400, 20, 440, 100)
    up_right = Pointing(np.array([100.0, 240.0]), np.array([0.83, -0.56]))
    sel = select_target(up_right, [cup, bottle], DIAG)
    assert sel.status is Status.SELECTED and sel.target is bottle


def test_two_objects_close_in_angle_is_unclear():
    a, b = det("cup", 400, 190, 440, 235), det("bottle", 400, 245, 440, 290)
    sel = select_target(pointing_right(), [a, b], DIAG)
    assert sel.status is Status.UNCLEAR and sel.target is None


def test_nothing_in_direction():
    sel = select_target(pointing_right(), [det("cup", 0, 0, 40, 40)], DIAG)
    assert sel.status is Status.NO_TARGET


def test_box_containing_fingertip_is_ignored():
    held = det("cell phone", 80, 220, 140, 260)
    cup = det("cup", 400, 200, 460, 280)
    sel = select_target(pointing_right(), [held, cup], DIAG)
    assert sel.target is cup


def test_stabilizer_commits_after_consecutive_frames():
    cup = det("cup", 400, 200, 460, 280)
    stab = TargetStabilizer()
    for _ in range(STABLE_FRAMES - 1):
        assert stab.update(Selection(Status.SELECTED, cup)) is None
    assert stab.update(Selection(Status.SELECTED, cup)) is cup
    assert stab.update(Selection(Status.UNCLEAR)) is None  # ambiguity drops the target


def test_stabilizer_tolerates_small_box_jitter():
    stab = TargetStabilizer()
    for i in range(STABLE_FRAMES):
        result = stab.update(Selection(Status.SELECTED, det("cup", 400 + i, 200, 460 + i, 280)))
    assert result is not None


def _hand(index_tip, curled=True):
    """Synthetic 21-point hand: wrist at bottom, index along +x from the knuckle."""
    lm = np.zeros((21, 2))
    lm[0] = [100, 300]
    lm[5], lm[6], lm[7] = [140, 250], [170, 250], [190, 250]
    lm[8] = index_tip
    for pip, tip in ((10, 12), (14, 16), (18, 20)):
        lm[pip] = [150, 260]
        lm[tip] = [120, 280] if curled else [150, 200]
    return lm


def test_classify_pointing_gesture():
    p = classify_pointing(_hand([210, 250]), DIAG)
    assert p is not None
    np.testing.assert_allclose(p.direction, [1.0, 0.0])


def test_open_hand_is_not_pointing():
    assert classify_pointing(_hand([210, 250], curled=False), DIAG) is None


def test_bent_index_is_not_pointing():
    assert classify_pointing(_hand([170, 280]), DIAG) is None


def test_head_mapping_directions():
    yaw, pitch = image_point_to_head_angles(np.array([640, 240]), 640, 480, mirrored=False)
    assert yaw < 0 and abs(pitch) < 1e-9  # camera's right -> Reachy turns right
    yaw_m, _ = image_point_to_head_angles(np.array([640, 240]), 640, 480, mirrored=True)
    assert yaw_m == -yaw
    _, pitch_down = image_point_to_head_angles(np.array([320, 480]), 640, 480, mirrored=False)
    assert pitch_down > 0
