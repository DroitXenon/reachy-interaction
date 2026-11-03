# Point, Look, and Name — Reachy Mini

Point at an object in front of your webcam. Reachy Mini (in the MuJoCo simulation) turns its
head toward the object and names it, both on screen and out loud. If it can't tell which object
you mean, it says so instead of guessing.

- Pointing: MediaPipe Hand Landmarker plus a geometric finger rule (`pointing.py`)
- Objects: pretrained YOLO11n, COCO classes, via ultralytics (`detection.py`)
- Target choice: deterministic ray/angle rule with ambiguity rejection and temporal stability (`selection.py`)
- Head: Reachy Mini SDK `goto_target`, with image position mapped to yaw/pitch (`robot.py`)
- Naming: on-screen banner and macOS `say` (`speech.py`)

## Setup

Requires Python 3.12 (MediaPipe does not support newer versions yet) and a webcam.

```bash
conda create -n reachy-interaction python=3.12 -y
conda activate reachy-interaction
pip install -r requirements.txt
```

Both models download automatically on first run: `models/hand_landmarker.task` and `yolo11n.pt`.

## Run

**Terminal 1: MuJoCo simulation** (on macOS, MuJoCo's viewer must run under `mjpython`):

```bash
conda activate reachy-interaction
mjpython -m reachy_mini.daemon.app.main --sim
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. If port 8000 is already in use, stop the other
Reachy daemon or the desktop app first.

**Terminal 2: the application**

```bash
conda activate reachy-interaction
python main.py
```

On macOS, the first run asks for camera permission for your terminal.

| Option | Effect |
|---|---|
| `--camera N` | OpenCV camera index (default 0) |
| `--no-robot` | Vision and speech only, without the simulator |
| `--mute` | No speech |
| `--no-mirror` | Show the raw camera image instead of a mirror view |
| `--yolo PATH` | Other ultralytics weights, e.g. `yolo11s.pt` for higher accuracy |

Keys: `q`/`Esc` quits, `r` resets the target and recentres the head.

## Using it

1. Put at least two COCO objects (e.g. cup, bottle, cell phone, book, scissors) on the table, a
   few hand-widths apart, so they're in the camera view.
2. Point at one with your index finger and the other fingers curled. Keep the finger visible
   side-on rather than aimed straight at the camera.
3. After about 0.3 s the box turns green, the banner shows `Looking at: <name>`, Reachy turns its
   head, and the name is spoken.
4. If you point between two objects, both boxes turn red and the banner shows `Unclear`. If that
   lasts, Reachy says "I'm not sure which one you mean".

Preview legend: blue = detected object, yellow = candidate / pointing ray, green = selected,
red = ambiguous candidates. Each candidate box shows its angle to the pointing ray.

## Tests

```bash
python -m pytest tests
```

These cover the ray/box test, target selection (hit, other object, ambiguity, nothing, held
object), the stabiliser, the gesture classifier and the head-angle mapping.

## Files

| File | Responsibility |
|---|---|
| `main.py` | CLI, camera loop, interaction logic (when to look, speak, recentre) |
| `pointing.py` | Hand Landmarker wrapper and pointing-gesture rule |
| `detection.py` | YOLO wrapper; filters out `person` and low scores |
| `selection.py` | Deterministic target selection and `TargetStabilizer` |
| `robot.py` | Image point → head yaw/pitch; non-blocking head controller |
| `speech.py` | Non-blocking text-to-speech |
| `draw.py` | Annotated preview |
| `report.md` | Short report describing the approach |
