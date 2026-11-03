"""Point, Look, and Name: point at an object, Reachy Mini looks at it and names it.

Run the MuJoCo simulation first (see README), then:
    python main.py
"""

import argparse
import time

import cv2
import numpy as np

from detection import ObjectDetector
from draw import GREEN, RED, WHITE, YELLOW, draw_detections, draw_hands, draw_status
from pointing import HandTracker
from selection import (
    STABLE_FRAMES,
    Selection,
    Status,
    TargetStabilizer,
    same_object,
    select_target,
)
from speech import Speaker

NEUTRAL_AFTER_S = 3.0  # return head to neutral after this long without a target
UNCLEAR_SPEAK_FRAMES = 2 * STABLE_FRAMES  # sustained ambiguity before asking the user
WINDOW = "Point, Look, and Name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--no-robot", action="store_true", help="run without the simulator")
    parser.add_argument("--mute", action="store_true", help="disable speech")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    parser.add_argument("--yolo", default="yolo11n.pt", help="ultralytics YOLO weights")
    return parser.parse_args()


def classify_frame(hands, detections, diag: float) -> Selection:
    """Turn one frame's hands and detections into a Selection."""
    if not hands:
        return Selection(Status.NO_HAND)
    pointers = [h.pointing for h in hands if h.pointing is not None]
    if not pointers:
        return Selection(Status.NOT_POINTING)
    if len(pointers) > 1:  # two pointing hands: we cannot tell which one counts
        return Selection(Status.UNCLEAR)
    return select_target(pointers[0], detections, diag)


def main() -> None:
    args = parse_args()
    mirrored = not args.no_mirror

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    tracker = HandTracker()
    detector = ObjectDetector(args.yolo)
    stabilizer = TargetStabilizer()
    speaker = Speaker(enabled=not args.mute)
    head = None
    if not args.no_robot:
        from robot import HeadController, image_point_to_head_angles

        head = HeadController()

    announced = None  # object Reachy is currently looking at / last named
    last_target_time = time.monotonic()
    unclear_streak = 0
    unclear_spoken = False
    fps = 0.0
    t_prev = time.monotonic()
    t0 = t_prev

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not available")
                break
            if mirrored:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            diag = float(np.hypot(w, h))
            now = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = tracker.process(rgb, int((now - t0) * 1000))
            detections = detector.detect(frame)
            selection = classify_frame(hands, detections, diag)
            committed = stabilizer.update(selection)

            # React to a newly committed target: look at it and name it.
            if committed is not None:
                last_target_time = now
                unclear_spoken = False
                if announced is None or not same_object(announced, committed):
                    announced = committed
                    if head is not None:
                        yaw, pitch = image_point_to_head_angles(committed.center, w, h, mirrored)
                        head.look(yaw, pitch)
                    speaker.say(f"That's a {committed.label}")
            elif announced is not None and now - last_target_time > NEUTRAL_AFTER_S:
                announced = None
                unclear_spoken = False
                if head is not None:
                    head.look(0.0, 0.0)

            # Sustained ambiguity: ask once per episode instead of guessing.
            unclear_streak = unclear_streak + 1 if selection.status is Status.UNCLEAR else 0
            if unclear_streak >= UNCLEAR_SPEAK_FRAMES and not unclear_spoken:
                unclear_spoken = True
                speaker.say("I'm not sure which one you mean")

            # Preview.
            draw_detections(frame, detections, selection, committed)
            draw_hands(frame, hands)
            if committed is not None:
                draw_status(frame, f"Looking at: {committed.label}", GREEN)
            elif selection.status is Status.SELECTED:
                draw_status(frame, f"Checking: {selection.target.label}...", YELLOW)
            elif selection.status is Status.UNCLEAR:
                draw_status(frame, selection.status.value, RED)
            else:
                draw_status(frame, selection.status.value, WHITE)

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt
            cv2.putText(frame, f"{fps:.0f} FPS", (w - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                stabilizer.reset()
                announced = None
                if head is not None:
                    head.look(0.0, 0.0)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        speaker.stop()
        if head is not None:
            head.close()


if __name__ == "__main__":
    main()
