"""Validated local storage for labeled eye-crop samples."""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.eye_cropper import EyeCrop


class CaptureLabel(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNCERTAIN = "UNCERTAIN"

    @property
    def directory(self) -> str:
        return self.value.lower()


LIGHTING_OPTIONS = ("bright", "normal", "dim")
POSE_OPTIONS = ("straight", "left", "right", "up", "down")
METADATA_FIELDS = (
    "image_path",
    "label",
    "subject_id",
    "session_id",
    "eye_side",
    "timestamp",
    "capture_id",
    "lighting",
    "head_pose",
)
DEFAULT_CAPTURE_COOLDOWN_MS = 200
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


@dataclass(frozen=True)
class CaptureResult:
    accepted: bool
    message: str
    capture_id: Optional[str] = None


class DatasetRecorder:
    """Save paired eye crops and append one metadata row per saved image."""

    def __init__(
        self,
        subject_id: str,
        session_id: str,
        lighting: str = "normal",
        head_pose: str = "straight",
        eye_size: int = 160,
        cooldown_ms: int = DEFAULT_CAPTURE_COOLDOWN_MS,
        dataset_root: Path | str = Path("data/raw"),
        project_root: Path | str = Path.cwd(),
    ) -> None:
        self.subject_id = self._validate_identifier(subject_id, "subject_id")
        self.session_id = self._validate_identifier(session_id, "session_id")
        if lighting not in LIGHTING_OPTIONS:
            raise ValueError(f"lighting must be one of: {', '.join(LIGHTING_OPTIONS)}")
        if head_pose not in POSE_OPTIONS:
            raise ValueError(f"head_pose must be one of: {', '.join(POSE_OPTIONS)}")
        if eye_size < 1:
            raise ValueError("eye_size must be positive")
        if cooldown_ms < 0:
            raise ValueError("cooldown_ms cannot be negative")

        self.lighting = lighting
        self.head_pose = head_pose
        self.eye_size = eye_size
        self.cooldown_seconds = cooldown_ms / 1000.0
        self.dataset_root = Path(dataset_root).resolve()
        self.project_root = Path(project_root).resolve()
        self.metadata_path = self.dataset_root / "metadata.csv"
        self._last_capture_at: Optional[float] = None
        self.counts = self._count_existing_images()
        self._prepare_storage()

    @staticmethod
    def _validate_identifier(value: str, field: str) -> str:
        if not value or not _IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError(
                f"{field} must be 1-32 letters, numbers, underscores, or hyphens "
                "and start with a letter or number"
            )
        return value

    def _prepare_storage(self) -> None:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        for label in CaptureLabel:
            (self.dataset_root / label.directory).mkdir(parents=True, exist_ok=True)

        if not self.metadata_path.exists():
            with self.metadata_path.open("x", newline="", encoding="utf-8") as metadata_file:
                csv.DictWriter(metadata_file, fieldnames=METADATA_FIELDS).writeheader()
        else:
            with self.metadata_path.open("r", newline="", encoding="utf-8") as metadata_file:
                header = next(csv.reader(metadata_file), [])
            if tuple(header) != METADATA_FIELDS:
                raise ValueError(f"Unexpected metadata.csv columns in {self.metadata_path}")

    def _count_existing_images(self) -> dict[str, int]:
        return {
            label.value: len(list((self.dataset_root / label.directory).glob("*.jpg")))
            for label in CaptureLabel
        }

    def next_capture_number(self) -> int:
        """Return the next number for this subject/session across all labels."""
        prefix = re.escape(f"{self.subject_id}_{self.session_id}_")
        pattern = re.compile(prefix + r"(?:left|right)_(\d+)\.jpg$")
        largest = 0
        for label in CaptureLabel:
            for path in (self.dataset_root / label.directory).glob("*.jpg"):
                match = pattern.fullmatch(path.name)
                if match:
                    largest = max(largest, int(match.group(1)))
        return largest + 1

    def record(
        self,
        label_value: str | CaptureLabel,
        left_crop: Optional[EyeCrop],
        right_crop: Optional[EyeCrop],
        now: Optional[float] = None,
    ) -> CaptureResult:
        """Validate and save one left/right crop pair for the given label."""
        try:
            label = label_value if isinstance(label_value, CaptureLabel) else CaptureLabel(label_value)
        except ValueError:
            return CaptureResult(False, f"Unknown label: {label_value}")

        if left_crop is None or right_crop is None:
            return CaptureResult(False, "Rejected: both eye crops must be detected")
        left_bytes = self._encode_crop(left_crop)
        right_bytes = self._encode_crop(right_crop)
        if left_bytes is None or right_bytes is None:
            return CaptureResult(False, f"Rejected: crops must be valid {self.eye_size}x{self.eye_size} images")

        captured_at = time.monotonic() if now is None else now
        if (
            self._last_capture_at is not None
            and captured_at - self._last_capture_at < self.cooldown_seconds
        ):
            remaining_ms = int((self.cooldown_seconds - (captured_at - self._last_capture_at)) * 1000) + 1
            return CaptureResult(False, f"Cooldown: wait {remaining_ms} ms")

        directory = self.dataset_root / label.directory
        while True:
            capture_id = f"{self.next_capture_number():06d}"
            base = f"{self.subject_id}_{self.session_id}"
            image_paths = (
                directory / f"{base}_left_{capture_id}.jpg",
                directory / f"{base}_right_{capture_id}.jpg",
            )
            created: list[Path] = []
            try:
                for path, image_data in zip(image_paths, (left_bytes, right_bytes)):
                    with path.open("xb") as image_file:
                        created.append(path)
                        image_file.write(image_data)
                        image_file.flush()
                rows = [
                    self._metadata_row(path, label, eye_side, capture_id)
                    for path, eye_side in zip(image_paths, ("left", "right"))
                ]
                with self.metadata_path.open("a+", newline="", encoding="utf-8") as metadata_file:
                    metadata_file.seek(0, 2)
                    metadata_offset = metadata_file.tell()
                    writer = csv.DictWriter(metadata_file, fieldnames=METADATA_FIELDS)
                    try:
                        writer.writerows(rows)
                        metadata_file.flush()
                    except OSError:
                        metadata_file.seek(metadata_offset)
                        metadata_file.truncate()
                        metadata_file.flush()
                        raise
            except FileExistsError:
                cleanup_error = self._remove_files(created)
                if cleanup_error:
                    return CaptureResult(False, f"Save collision; {cleanup_error}")
                continue
            except OSError as error:
                cleanup_error = self._remove_files(created)
                detail = f"; {cleanup_error}" if cleanup_error else ""
                return CaptureResult(False, f"Save failed: {error}{detail}")

            self._last_capture_at = captured_at
            self.counts[label.value] += 2
            return CaptureResult(True, f"Saved {label.value.lower()} pair {capture_id}", capture_id)

    def _encode_crop(self, crop: EyeCrop) -> Optional[bytes]:
        if not isinstance(crop, EyeCrop):
            return None
        image = crop.image
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.dtype != np.uint8
            or image.shape != (self.eye_size, self.eye_size, 3)
            or image.size == 0
        ):
            return None
        try:
            encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except (cv2.error, TypeError, ValueError):
            return None
        return buffer.tobytes() if encoded and buffer.size else None

    def _metadata_row(
        self,
        image_path: Path,
        label: CaptureLabel,
        eye_side: str,
        capture_id: str,
    ) -> dict[str, str]:
        try:
            stored_path = image_path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            stored_path = image_path.resolve().as_posix()
        return {
            "image_path": stored_path,
            "label": label.value,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "eye_side": eye_side,
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "capture_id": capture_id,
            "lighting": self.lighting,
            "head_pose": self.head_pose,
        }

    @staticmethod
    def _remove_files(paths: list[Path]) -> Optional[str]:
        errors = []
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(f"could not remove incomplete file {path}: {error}")
        return "; ".join(errors) if errors else None
