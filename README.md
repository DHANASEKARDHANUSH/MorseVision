# MorseVision

MorseVision is a local computer-vision project being built in stages toward assistive communication. Its current stage collects labeled eye images from the existing webcam, face-landmark, and eye-cropping pipeline.

## Current stage: Stage 2 — Eye-State Dataset Collection

The collector reuses the eye crops produced by `EyeCropper`, validates that both eyes are available, standardizes them to the configured square size, and saves one image per eye with session metadata. Collection and storage stay on this computer.

### Labels

- **OPEN**: eye is clearly open.
- **CLOSED**: eye is clearly closed.
- **UNCERTAIN**: ambiguous, blurry, occluded, poorly framed, or otherwise unsuitable sample.

`UNCERTAIN` is a quarantine/review class. It is not intended to be a third class in the later binary OPEN/CLOSED training data. Review it separately before deciding whether any sample is usable.

## Environment setup

This repository already has a `.venv` virtual environment. Activate that environment; do not create another one.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

The install command adds the project’s current runtime dependencies to the existing environment. A webcam and a desktop session with OpenCV GUI support are required.

## Start a collection session

From the project root with `.venv` active:

```powershell
python main.py --collect --subject S01 --session S02 --lighting normal --pose straight
```

Use anonymous subject and session identifiers. IDs may contain 1–32 letters, numbers, underscores, or hyphens, starting with a letter or number. The supported lighting values are `bright`, `normal`, and `dim`; supported pose values are `straight`, `left`, `right`, `up`, and `down`. These are manually assigned session-level descriptions, not automatically detected properties.

Optional capture settings:

```powershell
python main.py --collect --subject S02 --session S01 --lighting dim --pose left --eye-size 160 --cooldown-ms 200
```

`--eye-size` controls the square dimensions sent by the existing eye cropper. `--cooldown-ms` sets the minimum interval between accepted paired captures and defaults to 200 ms. The requested webcam resolution can also be set with `--width` and `--height`, and the camera device with `--camera`.

Without `--collect`, `python main.py` continues to run the Stage 1 preview without saving images.

## Controls and live display

- **O**: save the current pair as OPEN.
- **C**: save the current pair as CLOSED.
- **U**: save the current pair as UNCERTAIN.
- **Q** or **Esc**: quit.

Each key event requests one capture. The app saves only when a face and both valid eye crops are available. It displays the session IDs, lighting and pose, current label, saved image counts by class, face/eye status, FPS, controls, and the latest capture result. The second window continues to show the actual left and right crop images. A missing face or eye rejects the requested capture with a status message and does not crash the session.

## Dataset layout

```text
data/raw/
├── open/
├── closed/
├── uncertain/
└── metadata.csv
```

Each accepted paired capture creates two separate JPEGs, for example:

```text
data/raw/open/S01_S02_left_000001.jpg
data/raw/open/S01_S02_right_000001.jpg
```

Images are saved locally under their label directory at the configured eye size. `data/raw/metadata.csv` has one row per image, with `image_path`, `label`, `subject_id`, `session_id`, `eye_side`, `timestamp`, `capture_id`, `lighting`, and `head_pose`. The paired left/right rows share a capture ID. Counts shown in the app are image counts, so each accepted pair adds two to its label.

The recorder scans existing filenames for the current subject and session at startup, continues with the next available capture number, and opens image paths exclusively. It will not overwrite an existing image. The cooldown reduces accidental rapid repeats; it does not rebalance or delete data. Generated data, including `metadata.csv`, is ignored by Git.

## Recommended collection protocol

An initial target of roughly **8–12 subjects** and **3–5 sessions per subject** is a useful starting point, not a hard quota. Collect balanced OPEN and CLOSED examples while prioritizing diversity and subject separation over raw image count.

Across sessions, vary bright, normal, and dim lighting; straight and slight left, right, up, or down poses; camera distance and framing; and glasses/no glasses where applicable. Include different participants and natural variation in eye shape, facial structure, and skin tone. Use a new session ID when collecting a distinct session condition. Avoid collecting thousands of near-identical frames. Normal blinks may be useful for later temporal testing, but should not be blindly labeled and dumped into the OPEN/CLOSED training data.

Future evaluation should split by subject so that images of one person do not appear in both training and evaluation sets. This collector does not create any split.

## Privacy

Use anonymous subject IDs; do not put names, email addresses, phone numbers, or other personal information in IDs or filenames. The collector does not upload images or metadata to a cloud service. Keep collection local and obtain appropriate consent from participants.

## Tests

The deterministic tests cover crop geometry, dataset image validation, label/ID validation, unique numbering, metadata rows, counters, and cooldown. Run them from the project root in the active `.venv`:

```powershell
python -m pip install pytest
python -m pytest
```

No fake webcam tests are used.

## Current limitations

- CNN classification is not implemented.
- Morse decoding is not implemented.
- Temporal modeling is not implemented.
- Data augmentation is not implemented.
- Automated subject-aware splitting is not implemented.
- UNCERTAIN samples are stored for review and should not be blindly used as a training class.
- Lighting and pose are session-level labels entered by the collector.

## Roadmap

1. **Stage 3:** Dataset inspection and preprocessing.
2. **Stage 4:** CNN training.
3. **Stage 5:** CNN evaluation.
4. **Stage 6:** Real-time CNN inference.
5. **Stage 7:** Temporal eye-state analysis.
6. **Stage 8:** Morse decoding.
7. **Stage 9:** ONNX deployment and optimization.
