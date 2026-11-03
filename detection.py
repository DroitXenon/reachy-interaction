"""Object detection with a pretrained YOLO model (COCO classes) via ultralytics."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "yolo11n.pt"  # downloaded automatically by ultralytics on first use
SCORE_THRESHOLD = 0.4
IGNORED_LABELS = {"person"}  # the user is always in frame; never a pointing target


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: np.ndarray  # (4,) x1, y1, x2, y2 in pixels

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


class ObjectDetector:
    def __init__(self, model: str = DEFAULT_MODEL, score_threshold: float = SCORE_THRESHOLD):
        self._model = YOLO(model)
        self._threshold = score_threshold

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        result = self._model(bgr, conf=self._threshold, verbose=False)[0]
        names = result.names
        detections = []
        for xyxy, score, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            label = names[cls]
            if label not in IGNORED_LABELS:
                detections.append(Detection(label, float(score), xyxy.astype(np.float64)))
        return detections
