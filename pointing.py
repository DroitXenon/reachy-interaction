"""Pointing-gesture detection with MediaPipe Hand Landmarker.

The gesture rule works on 2D image landmarks (pixels):
  * the index finger is straight (joint chain almost a line), and
  * at least two of middle/ring/pinky are curled (tip closer to the wrist than its PIP joint).
The pointing ray starts at the index tip and follows the knuckle->tip direction.
"""

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"

# MediaPipe landmark indices.
WRIST = 0
INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, TIP
OTHER_FINGERS = ((10, 12), (14, 16), (18, 20))  # (PIP, TIP) for middle, ring, pinky

# Gesture thresholds (see report for rationale).
INDEX_STRAIGHTNESS_MIN = 0.85  # |tip - mcp| / joint-chain length; 1.0 = perfectly straight
CURL_RATIO_MAX = 1.1  # finger curled if dist(tip, wrist) < 1.1 * dist(pip, wrist)
MIN_CURLED_FINGERS = 2  # of middle/ring/pinky; tolerates a loose pinky
MIN_INDEX_LENGTH_FRAC = 0.04  # of image diagonal; rejects pointing straight at the camera


@dataclass(frozen=True)
class Pointing:
    origin: np.ndarray  # (2,) index tip in pixels
    direction: np.ndarray  # (2,) unit vector in image coordinates


@dataclass(frozen=True)
class HandResult:
    landmarks: np.ndarray  # (21, 2) pixel coordinates
    pointing: Pointing | None


def classify_pointing(landmarks: np.ndarray, image_diag: float) -> Pointing | None:
    """Return a Pointing if the hand landmarks form a pointing gesture, else None."""
    mcp, pip, dip, tip = (landmarks[i] for i in INDEX)
    chain = np.linalg.norm(pip - mcp) + np.linalg.norm(dip - pip) + np.linalg.norm(tip - dip)
    span = np.linalg.norm(tip - mcp)
    if chain <= 0 or span < MIN_INDEX_LENGTH_FRAC * image_diag:
        return None
    if span / chain < INDEX_STRAIGHTNESS_MIN:
        return None

    wrist = landmarks[WRIST]
    curled = sum(
        np.linalg.norm(landmarks[t] - wrist) < CURL_RATIO_MAX * np.linalg.norm(landmarks[p] - wrist)
        for p, t in OTHER_FINGERS
    )
    if curled < MIN_CURLED_FINGERS:
        return None

    return Pointing(origin=tip.copy(), direction=(tip - mcp) / span)


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Wraps MediaPipe Hand Landmarker in VIDEO mode."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, num_hands: int = 2):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandResult]:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = rgb.shape[:2]
        diag = float(np.hypot(w, h))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands = []
        for hand in result.hand_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float64)
            hands.append(HandResult(landmarks=pts, pointing=classify_pointing(pts, diag)))
        return hands

    def close(self) -> None:
        self._landmarker.close()
