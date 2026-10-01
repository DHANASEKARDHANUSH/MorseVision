"""Face Mesh wrapper that exposes image-space face and eye landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.eye_cropper import BoundingBox, Point

# MediaPipe Face Mesh contour landmarks, using anatomical left/right naming.
LEFT_EYE_INDICES = (362, 385, 387, 263, 373, 380, 386, 388, 466, 374, 398, 382)
RIGHT_EYE_INDICES = (33, 160, 158, 133, 153, 144, 159, 161, 246, 145, 173, 154)


@dataclass(frozen=True)
class FaceObservation:
    """Landmarks for the primary face, expressed in pixel coordinates."""

    face_box: BoundingBox
    left_eye: tuple[Point, ...]
    right_eye: tuple[Point, ...]


class FaceDetector:
    """Detect up to several faces and return landmarks for the largest one."""

    def __init__(
        self,
        max_faces: int = 3,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> Optional[FaceObservation]:
        """Return landmarks for the largest visible face in a BGR frame."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return None

        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return None
        result = self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = result.multi_face_landmarks or []
        if not faces:
            return None

        # Select the primary face by its clamped bounding-box area.
        def face_area(points: object) -> int:
            xs = [min(width - 1, max(0, round(p.x * width))) for p in points]
            ys = [min(height - 1, max(0, round(p.y * height))) for p in points]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        primary = max(faces, key=lambda face: face_area(face.landmark))
        landmarks = primary.landmark
        pixel_points = [
            Point(min(width - 1, max(0, round(p.x * width))),
                  min(height - 1, max(0, round(p.y * height))))
            for p in landmarks
        ]
        face_xs = [point.x for point in pixel_points]
        face_ys = [point.y for point in pixel_points]
        box = BoundingBox(
            min(face_xs), min(face_ys),
            max(face_xs) - min(face_xs) + 1,
            max(face_ys) - min(face_ys) + 1,
        )
        return FaceObservation(
            face_box=box,
            left_eye=tuple(pixel_points[i] for i in LEFT_EYE_INDICES),
            right_eye=tuple(pixel_points[i] for i in RIGHT_EYE_INDICES),
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._mesh.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
