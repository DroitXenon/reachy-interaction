"""Reachy Mini head control (MuJoCo simulation via the Reachy Mini daemon).

Head-movement mapping: Reachy is imagined sitting at the webcam, looking at the same
scene. A target's position in the camera image is turned into a head yaw/pitch with a
linear angle-per-pixel mapping from the webcam's field of view.
"""

import queue
import threading

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

CAMERA_HFOV_DEG = 60.0  # typical laptop webcam
CAMERA_VFOV_DEG = 45.0
MAX_YAW_DEG = 35.0
MAX_PITCH_DEG = 20.0
MOVE_DURATION_S = 0.6


def image_point_to_head_angles(
    point: np.ndarray, width: int, height: int, mirrored: bool
) -> tuple[float, float]:
    """Map an image point to (yaw, pitch) in degrees.

    yaw > 0 turns the head to Reachy's left; pitch > 0 tilts it down.
    In the raw camera image, the right side is the camera's right, so it needs negative
    yaw. A mirrored preview flips x first.
    """
    x = point[0] / width
    if mirrored:
        x = 1.0 - x
    y = point[1] / height
    yaw = -(x - 0.5) * CAMERA_HFOV_DEG
    pitch = (y - 0.5) * CAMERA_VFOV_DEG
    return (
        float(np.clip(yaw, -MAX_YAW_DEG, MAX_YAW_DEG)),
        float(np.clip(pitch, -MAX_PITCH_DEG, MAX_PITCH_DEG)),
    )


class HeadController:
    """Sends head moves from a background thread so the camera loop never blocks.

    Only the most recent request is kept: if a new target arrives while the head is
    moving, older pending requests are dropped.
    """

    def __init__(self):
        self._mini = ReachyMini(media_backend="no_media")
        self._requests: queue.Queue[tuple[float, float] | None] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.look(0.0, 0.0)

    def look(self, yaw_deg: float, pitch_deg: float) -> None:
        self._replace((yaw_deg, pitch_deg))

    def close(self) -> None:
        self.look(0.0, 0.0)
        self._replace(None)
        self._thread.join(timeout=2.0)
        # Disconnect explicitly; leaving it to ReachyMini.__del__ at interpreter exit hangs.
        self._mini.__exit__(None, None, None)

    def _replace(self, item: tuple[float, float] | None) -> None:
        try:
            self._requests.get_nowait()
        except queue.Empty:
            pass
        self._requests.put_nowait(item)

    def _run(self) -> None:
        while (item := self._requests.get()) is not None:
            yaw, pitch = item
            pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
            self._mini.goto_target(head=pose, duration=MOVE_DURATION_S)
