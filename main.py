"""Stage 1 webcam preview for face and eye-region detection."""

from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np

from src.camera import Camera
from src.dataset_recorder import (
    DEFAULT_CAPTURE_COOLDOWN_MS,
    CaptureLabel,
    DatasetRecorder,
)
from src.eye_cropper import EyeCrop, EyeCropper
from src.face_detector import FaceDetector
from src.utils import FPSCounter, draw_box, draw_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview face landmarks and eye crops.")
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--eye-size", type=int, default=160, help="square eye crop output size")
    parser.add_argument("--padding", type=float, default=0.25, help="eye box padding ratio")
    parser.add_argument("--collect", action="store_true", help="enable labeled eye dataset collection")
    parser.add_argument("--subject", help="anonymous subject ID, e.g. S02 (required with --collect)")
    parser.add_argument("--session", help="session ID, e.g. S03 (required with --collect)")
    parser.add_argument(
        "--lighting", choices=("bright", "normal", "dim"), default="normal",
        help="manual session lighting label",
    )
    parser.add_argument(
        "--pose", choices=("straight", "left", "right", "up", "down"), default="straight",
        help="manual session head-pose label",
    )
    parser.add_argument(
        "--cooldown-ms", type=int, default=DEFAULT_CAPTURE_COOLDOWN_MS,
        help="minimum time between accepted capture pairs",
    )
    args = parser.parse_args()
    if args.collect and (not args.subject or not args.session):
        parser.error("--collect requires non-blank --subject and --session values")
    if args.eye_size < 1:
        parser.error("--eye-size must be positive")
    if args.padding < 0:
        parser.error("--padding cannot be negative")
    if args.cooldown_ms < 0:
        parser.error("--cooldown-ms cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    recorder: Optional[DatasetRecorder] = None
    if args.collect:
        try:
            recorder = DatasetRecorder(
                subject_id=args.subject,
                session_id=args.session,
                lighting=args.lighting,
                head_pose=args.pose,
                eye_size=args.eye_size,
                cooldown_ms=args.cooldown_ms,
            )
        except (OSError, ValueError) as error:
            print(f"Dataset setup failed: {error}")
            return 1

    detector = FaceDetector()
    try:
        camera = Camera(args.camera, args.width, args.height)
    except RuntimeError as error:
        detector.close()
        print(f"Camera unavailable: {error}")
        return 1

    cropper = EyeCropper(output_size=args.eye_size, padding_ratio=args.padding)
    fps = FPSCounter()
    if recorder:
        print(f"Collecting for subject {recorder.subject_id}, session {recorder.session_id}. Press Q to quit.")
    else:
        print("Press Q or Esc to quit.")

    current_label = "IDLE"
    capture_status = "Ready" if recorder else ""

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Camera frame read failed; stopping preview.")
                break

            observation = detector.detect(frame)
            fps.update()
            display = frame.copy()
            left_crop = right_crop = None

            if observation is None:
                draw_label(display, "Face: NOT DETECTED", (12, 28), (0, 165, 255))
                draw_label(display, "Left Eye: NOT DETECTED", (12, 56), (0, 165, 255))
                draw_label(display, "Right Eye: NOT DETECTED", (12, 84), (0, 165, 255))
            else:
                draw_box(display, observation.face_box, (255, 160, 30), "Face")
                left_crop = cropper.crop(frame, observation.left_eye)
                right_crop = cropper.crop(frame, observation.right_eye)
                draw_label(display, "Face: DETECTED", (12, 28))
                draw_label(
                    display,
                    f"Left Eye: {'DETECTED' if left_crop else 'NOT DETECTED'}",
                    (12, 56),
                    (40, 220, 40) if left_crop else (0, 165, 255),
                )
                draw_label(
                    display,
                    f"Right Eye: {'DETECTED' if right_crop else 'NOT DETECTED'}",
                    (12, 84),
                    (40, 220, 40) if right_crop else (0, 165, 255),
                )
                if left_crop:
                    draw_box(display, left_crop.box, (60, 230, 60), "Left eye")
                if right_crop:
                    draw_box(display, right_crop.box, (0, 220, 255), "Right eye")

            if recorder:
                _draw_collection_overlay(
                    display, recorder, current_label, capture_status,
                    observation is not None, left_crop, right_crop, fps.value,
                )
            else:
                draw_label(display, f"FPS: {fps.value:.1f}", (12, 112), (255, 255, 255))
            cv2.imshow("Morse Vision - Face and Eyes", display)
            eye_view = _compose_eye_view(left_crop, right_crop, args.eye_size)
            cv2.imshow("Eye crops (left | right)", eye_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if recorder:
                label_by_key = {
                    ord("o"): CaptureLabel.OPEN,
                    ord("O"): CaptureLabel.OPEN,
                    ord("c"): CaptureLabel.CLOSED,
                    ord("C"): CaptureLabel.CLOSED,
                    ord("u"): CaptureLabel.UNCERTAIN,
                    ord("U"): CaptureLabel.UNCERTAIN,
                }
                if key in label_by_key:
                    label = label_by_key[key]
                    current_label = label.value
                    result = recorder.record(label, left_crop, right_crop)
                    capture_status = result.message
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


def _draw_collection_overlay(
    frame: np.ndarray,
    recorder: DatasetRecorder,
    current_label: str,
    capture_status: str,
    face_detected: bool,
    left_crop: Optional[EyeCrop],
    right_crop: Optional[EyeCrop],
    fps: float,
) -> None:
    """Draw the session details, collection counts, statuses, and controls."""
    lines = (
        ("MorseVision - Dataset Collection", (255, 255, 255)),
        (f"Subject: {recorder.subject_id}    Session: {recorder.session_id}", (255, 255, 255)),
        (f"Lighting: {recorder.lighting}    Pose: {recorder.head_pose}", (255, 255, 255)),
        (f"Current Label: {current_label}", (80, 230, 80)),
        (f"OPEN: {recorder.counts['OPEN']}  CLOSED: {recorder.counts['CLOSED']}  UNCERTAIN: {recorder.counts['UNCERTAIN']}", (255, 255, 255)),
        (f"Face: {'DETECTED' if face_detected else 'NOT DETECTED'}", (80, 230, 80) if face_detected else (0, 165, 255)),
        (f"Left Eye: {'DETECTED' if left_crop else 'NOT DETECTED'}", (80, 230, 80) if left_crop else (0, 165, 255)),
        (f"Right Eye: {'DETECTED' if right_crop else 'NOT DETECTED'}", (80, 230, 80) if right_crop else (0, 165, 255)),
        (f"FPS: {fps:.1f}    {capture_status}", (255, 255, 255)),
        ("Controls: O = Open  C = Closed  U = Uncertain  Q = Quit", (255, 255, 255)),
    )
    for index, (text, color) in enumerate(lines):
        draw_label(frame, text, (12, 24 + index * 27), color)


def _compose_eye_view(
    left_crop: Optional[EyeCrop], right_crop: Optional[EyeCrop], size: int
) -> np.ndarray:
    """Show both crop slots, including a placeholder when a crop is absent."""
    tiles = []
    for crop, label in ((left_crop, "Left eye"), (right_crop, "Right eye")):
        tile = crop.image.copy() if crop is not None else np.zeros((size, size, 3), dtype=np.uint8)
        draw_label(tile, label if crop is not None else f"{label}: unavailable", (8, 22),
                   (80, 230, 80) if crop is not None else (180, 180, 180))
        tiles.append(tile)
    return cv2.hconcat(tiles)


if __name__ == "__main__":
    raise SystemExit(main())
