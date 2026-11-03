"""Deterministic rule for choosing which detected object the user is pointing at.

Rule, per frame:
  1. Discard boxes that contain the fingertip (the hand itself or a held object)
     and boxes whose centre lies behind the fingertip along the ray.
  2. For each remaining box compute theta, the angle between the pointing ray and the
     vector from the fingertip to the box centre.
  3. A box is a candidate if the ray passes through the box (padded slightly) or theta
     is small.
  4. The candidate with the smallest theta wins, unless the runner-up is within
     AMBIGUITY_MARGIN_DEG: then the frame is UNCLEAR and nothing is selected.

Over time, TargetStabilizer only commits a target once the same object has won for
several consecutive frames.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from detection import Detection
from pointing import Pointing

MAX_ANGLE_DEG = 12.0  # ray may miss the box but still be "close enough" in angle
AMBIGUITY_MARGIN_DEG = 6.0  # runner-up this close to the best -> refuse to guess
BOX_PADDING_FRAC = 0.03  # of image diagonal, added on every side for the ray-box test
STABLE_FRAMES = 6  # consecutive frames the same object must win before committing
SAME_OBJECT_IOU = 0.3  # boxes in consecutive frames with this IoU are the same object


class Status(Enum):
    NO_HAND = "No hand"
    NOT_POINTING = "Not pointing"
    NO_TARGET = "Pointing at nothing"
    UNCLEAR = "Unclear - which one?"
    SELECTED = "Selected"


@dataclass(frozen=True)
class Candidate:
    detection: Detection
    angle_deg: float


@dataclass(frozen=True)
class Selection:
    status: Status
    target: Detection | None = None
    candidates: tuple[Candidate, ...] = ()


def ray_hits_box(origin: np.ndarray, direction: np.ndarray, box: np.ndarray) -> bool:
    """Slab test: does the forward ray origin + t*direction (t >= 0) intersect the box?"""
    t_near, t_far = 0.0, np.inf
    for axis in range(2):
        lo, hi = box[axis], box[axis + 2]
        if abs(direction[axis]) < 1e-9:
            if not lo <= origin[axis] <= hi:
                return False
            continue
        t1 = (lo - origin[axis]) / direction[axis]
        t2 = (hi - origin[axis]) / direction[axis]
        t_near = max(t_near, min(t1, t2))
        t_far = min(t_far, max(t1, t2))
    return t_near <= t_far


def _contains(box: np.ndarray, point: np.ndarray) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def angle_to(pointing: Pointing, point: np.ndarray) -> float:
    v = point - pointing.origin
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return 180.0
    cos = float(np.clip(np.dot(v / norm, pointing.direction), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def select_target(
    pointing: Pointing, detections: list[Detection], image_diag: float
) -> Selection:
    pad = BOX_PADDING_FRAC * image_diag
    candidates = []
    for det in detections:
        if _contains(det.box, pointing.origin):
            continue
        theta = angle_to(pointing, det.center)
        if theta >= 90.0:  # behind the fingertip
            continue
        padded = det.box + np.array([-pad, -pad, pad, pad])
        if ray_hits_box(pointing.origin, pointing.direction, padded) or theta <= MAX_ANGLE_DEG:
            candidates.append(Candidate(det, theta))

    candidates.sort(key=lambda c: c.angle_deg)
    ranked = tuple(candidates)
    if not ranked:
        return Selection(Status.NO_TARGET)
    if len(ranked) > 1 and ranked[1].angle_deg - ranked[0].angle_deg < AMBIGUITY_MARGIN_DEG:
        return Selection(Status.UNCLEAR, candidates=ranked)
    return Selection(Status.SELECTED, ranked[0].detection, ranked)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_object(a: Detection, b: Detection) -> bool:
    return a.label == b.label and iou(a.box, b.box) >= SAME_OBJECT_IOU


class TargetStabilizer:
    """Commits a target only after it wins STABLE_FRAMES frames in a row.

    Any frame without a selection (unclear, no target, no gesture) resets the streak,
    so a committed target is dropped as soon as pointing stops being clear.
    """

    def __init__(self, stable_frames: int = STABLE_FRAMES):
        self._needed = stable_frames
        self._candidate: Detection | None = None
        self._streak = 0
        self.committed: Detection | None = None

    def update(self, selection: Selection) -> Detection | None:
        target = selection.target if selection.status is Status.SELECTED else None
        if target is None:
            self.reset()
            return None
        if self._candidate is not None and same_object(self._candidate, target):
            self._streak += 1
        else:
            self._streak = 1
        self._candidate = target
        if self._streak >= self._needed:
            self.committed = target
        elif self.committed is not None and not same_object(self.committed, target):
            self.committed = None
        return self.committed

    def reset(self) -> None:
        self._candidate = None
        self._streak = 0
        self.committed = None
