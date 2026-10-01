# Morse Vision

A local Python computer-vision project inspired by [Hello-Morse-OpenCV](https://github.com/raghavpatnecha/Hello-Morse-OpenCV). This repository is being built in stages as a foundation for an assistive communication system.

## Stage 1

Stage 1 opens a webcam, detects faces and facial landmarks with MediaPipe Face Mesh, selects the largest detected face, extracts padded left and right eye crops, and displays those crops alongside the camera view. The camera view includes face and eye boxes, detection status, and rolling FPS. Press **Q** or **Esc** to exit.

**CNN-based eye-state classification and Morse decoding are not implemented.** Stage 1 does not classify eyes as open or closed, record a dataset, or interpret blinks.

## Requirements

- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- A webcam
- A desktop session with OpenCV GUI support

MediaPipe 0.10.21 is pinned because this stage uses its bundled `solutions.face_mesh` API. The project does not require dlib or a separately downloaded landmark model.

## Installation

From the project directory, create and activate a virtual environment in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell does not find Python 3.11, use `py -3.10` or `py -3.12` if installed. On macOS or Linux, create/activate the environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then run the two pip commands above.

## Run

```powershell
python main.py
```

Optional camera and crop settings:

```powershell
python main.py --camera 1 --width 640 --height 480 --eye-size 160 --padding 0.25
```

Camera index 0 is the default. OpenCV treats the requested capture resolution as a preference; a camera or driver may provide a different size. A window shows the live frame with the primary face and eye boxes. A second window shows the left and right eye image arrays resized to `--eye-size`. When no face or eye crop is available, the status and crop preview indicate that. If opening or reading from the webcam fails, the program reports the issue and exits cleanly.

## Project structure

```text
data/
  raw/open/             # Reserved for a later dataset stage
  raw/closed/           # Reserved for a later dataset stage
  processed/            # Reserved for later preprocessing
models/
  checkpoints/          # Reserved for future model training
  exported/             # Reserved for future model export
src/
  camera.py             # Webcam capture and cleanup
  face_detector.py      # Face Mesh inference and eye landmark selection
  eye_cropper.py        # Padded eye boxes and crop image arrays
  utils.py              # FPS and drawing helpers
main.py                 # Webcam application entry point
requirements.txt        # Stage 1 runtime dependencies
tests/                  # Non-camera logic tests
```

## Current limitations

- Face Mesh landmarks are used to find eye regions; eye state is not determined.
- The largest detected face is processed. Multi-person interaction is out of scope.
- Crop boxes are axis-aligned and do not rotate to compensate for head tilt.
- Camera availability and GUI behavior depend on the host operating system, camera driver, and desktop session.
- The crop preview size can be configured, but lighting, framing, and focus affect landmark quality.

## Planned stages

Later stages may add an eye-state dataset recorder, a small CNN and training workflow, ONNX export and OpenCV DNN inference, confidence and temporal analysis, calibration, Morse decoding, evaluation/optimization, and optional offline speech. These features are intentionally absent from Stage 1.

## Basic tests

The non-camera tests cover crop bounding/clipping, crop output validation, and FPS calculation. Run them from the project directory after installing the runtime dependencies and pytest:

```powershell
python -m pip install pytest
python -m pytest
```
