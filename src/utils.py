"""Small display and timing helpers."""

from __future__ import annotations

from collections import deque
from time import monotonic
from typing import Deque, Optional

import cv2
import numpy as np

from src.eye_cropper import BoundingBox


class FPSCounter:
    """Calculate average FPS over a short monotonic-time window."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._times: Deque[float] = deque(maxlen=window_size)

    def update(self, timestamp: Optional[float] = None) -> float:
        """Record a frame time and return the current rolling FPS."""
        now = monotonic() if timestamp is None else timestamp
        if self._times and now <= self._times[-1]:
            return self.value
        self._times.append(now)
        return self.value

    @property
    def value(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (40, 220, 40),
) -> None:
    """Draw readable text with a dark outline."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)


def draw_box(
    frame: np.ndarray,
    box: BoundingBox,
    color: tuple[int, int, int],
    label: Optional[str] = None,
) -> None:
    """Draw a bounding box and an optional label on a frame."""
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.width - 1, box.y + box.height - 1), color, 2)
    if label:
        draw_label(frame, label, (box.x, max(22, box.y - 7)), color)
