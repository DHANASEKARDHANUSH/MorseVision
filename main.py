"""Stage 1 webcam preview for face and eye-region detection."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.camera import Camera
from src.eye_cropper import EyeCropper
from src.face_detector import FaceDetector
from src.utils import FPSCounter, draw_box, draw_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview face landmarks and eye crops.")
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--eye-size", type=int, default=160, help="square eye crop output size")
    parser.add_argument("--padding", type=float, default=0.25, help="eye box padding ratio")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detector = FaceDetector()
    try:
        camera = Camera(args.camera, args.width, args.height)
    except RuntimeError as error:
        detector.close()
        print(f"Camera unavailable: {error}")
        return 1

    cropper = EyeCropper(output_size=args.eye_size, padding_ratio=args.padding)
    fps = FPSCounter()
    print("Press Q or Esc to quit.")

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

            draw_label(display, f"FPS: {fps.value:.1f}", (12, 112), (255, 255, 255))
            cv2.imshow("Morse Vision - Face and Eyes", display)
            eye_view = _compose_eye_view(left_crop, right_crop, args.eye_size)
            cv2.imshow("Eye crops (left | right)", eye_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


def _compose_eye_view(left_crop, right_crop, size: int) -> np.ndarray:
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
