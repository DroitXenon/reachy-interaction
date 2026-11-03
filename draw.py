"""Annotated camera preview: hand skeleton, pointing ray, boxes, and status banner."""

import cv2
import numpy as np

from detection import Detection
from pointing import HandResult
from selection import Selection, Status

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
GREY = (160, 160, 160)
BLUE = (255, 140, 0)
YELLOW = (0, 220, 255)
GREEN = (0, 200, 0)
RED = (0, 0, 230)
WHITE = (255, 255, 255)


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def draw_hands(img: np.ndarray, hands: list[HandResult]) -> None:
    diag = float(np.hypot(*img.shape[:2]))
    for hand in hands:
        lm = hand.landmarks
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, _pt(lm[a]), _pt(lm[b]), GREY, 2)
        for p in lm:
            cv2.circle(img, _pt(p), 3, WHITE, -1)
        if hand.pointing is not None:
            end = hand.pointing.origin + hand.pointing.direction * diag
            cv2.arrowedLine(img, _pt(hand.pointing.origin), _pt(end), YELLOW, 3, tipLength=0.03)


def draw_detections(
    img: np.ndarray,
    detections: list[Detection],
    selection: Selection | None,
    committed: Detection | None,
) -> None:
    candidates = {id(c.detection): c for c in selection.candidates} if selection else {}
    for det in detections:
        cand = candidates.get(id(det))
        if committed is not None and det is committed:
            color, thickness = GREEN, 4
        elif cand is not None and selection.status is Status.UNCLEAR:
            color, thickness = RED, 3
        elif cand is not None:
            color, thickness = YELLOW, 2
        else:
            color, thickness = BLUE, 2
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = f"{det.label} {det.score:.2f}"
        if cand is not None:
            text += f" | {cand.angle_deg:.0f}deg"
        cv2.putText(img, text, (x1 + 3, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_status(img: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
