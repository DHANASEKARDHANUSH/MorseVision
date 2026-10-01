"""OpenCV camera capture and cleanup."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class Camera:
    """Small context-managed wrapper around ``cv2.VideoCapture``."""

    def __init__(
        self,
        device_index: int = 0,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
    ) -> None:
        self._capture = cv2.VideoCapture(device_index)
        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera device {device_index}.")

    def read(self) -> Optional[np.ndarray]:
        """Read one BGR frame, returning ``None`` when capture fails."""
        ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    def release(self) -> None:
        """Release the camera device."""
        self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
