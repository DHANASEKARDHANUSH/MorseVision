"""Eye landmark bounds and CNN-ready image crop extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class EyeCrop:
    image: np.ndarray
    box: BoundingBox


class EyeCropper:
    """Build padded eye regions and resize valid crops to a configurable size."""

    def __init__(self, output_size: Optional[int] = 160, padding_ratio: float = 0.25) -> None:
        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be positive or None")
        if padding_ratio < 0:
            raise ValueError("padding_ratio cannot be negative")
        self.output_size = output_size
        self.padding_ratio = padding_ratio

    def bounding_box(
        self, points: Sequence[Point], frame_shape: Sequence[int]
    ) -> Optional[BoundingBox]:
        """Calculate a clipped, padded box around eye landmarks."""
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0 or not points:
            return None
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        pad_x = round((max(xs) - min(xs) + 1) * self.padding_ratio)
        pad_y = round((max(ys) - min(ys) + 1) * self.padding_ratio)
        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(frame_width, max(xs) + pad_x + 1)
        y2 = min(frame_height, max(ys) + pad_y + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

    def crop(self, frame: np.ndarray, points: Sequence[Point]) -> Optional[EyeCrop]:
        """Return an actual eye image array and its source-frame box."""
        if frame is None or frame.ndim < 2 or frame.size == 0:
            return None
        box = self.bounding_box(points, frame.shape)
        if box is None:
            return None
        image = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        if image.size == 0:
            return None
        if self.output_size is not None:
            image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        return EyeCrop(image=image, box=box)
