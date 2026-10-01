import csv

import cv2
import numpy as np
import pytest

from src.dataset_recorder import CaptureLabel, DatasetRecorder, METADATA_FIELDS
from src.eye_cropper import BoundingBox, EyeCrop


def make_crop(size=16):
    return EyeCrop(
        image=np.full((size, size, 3), 120, dtype=np.uint8),
        box=BoundingBox(0, 0, size, size),
    )


def make_recorder(tmp_path, **overrides):
    options = {
        "subject_id": "S02",
        "session_id": "S03",
        "eye_size": 16,
        "dataset_root": tmp_path / "data" / "raw",
        "project_root": tmp_path,
    }
    options.update(overrides)
    return DatasetRecorder(**options)


def test_capture_writes_two_images_and_metadata_rows(tmp_path):
    recorder = make_recorder(tmp_path, lighting="dim", head_pose="left")

    result = recorder.record(CaptureLabel.OPEN, make_crop(), make_crop(), now=10.0)

    assert result.accepted
    assert result.capture_id == "000001"
    saved_paths = sorted((tmp_path / "data/raw/open").glob("*.jpg"))
    assert [path.name for path in saved_paths] == [
        "S02_S03_left_000001.jpg",
        "S02_S03_right_000001.jpg",
    ]
    assert all(cv2.imread(str(path)).shape == (16, 16, 3) for path in saved_paths)
    with (tmp_path / "data/raw/metadata.csv").open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    assert tuple(rows[0]) == METADATA_FIELDS
    assert [row["eye_side"] for row in rows] == ["left", "right"]
    assert all(row["label"] == "OPEN" for row in rows)
    assert all(row["subject_id"] == "S02" and row["session_id"] == "S03" for row in rows)
    assert all(row["lighting"] == "dim" and row["head_pose"] == "left" for row in rows)
    assert recorder.counts["OPEN"] == 2


def test_numbering_resumes_and_is_shared_across_label_directories(tmp_path):
    recorder = make_recorder(tmp_path)
    first = recorder.record(CaptureLabel.OPEN, make_crop(), make_crop(), now=1.0)
    second = recorder.record(CaptureLabel.CLOSED, make_crop(), make_crop(), now=1.3)

    assert first.capture_id == "000001"
    assert second.capture_id == "000002"
    restarted = make_recorder(tmp_path)
    assert restarted.next_capture_number() == 3


def test_preexisting_filename_is_preserved(tmp_path):
    recorder = make_recorder(tmp_path)
    existing = tmp_path / "data/raw/open/S02_S03_left_000001.jpg"
    existing.write_bytes(b"keep existing file")

    result = recorder.record(CaptureLabel.OPEN, make_crop(), make_crop(), now=1.0)

    assert result.capture_id == "000002"
    assert existing.read_bytes() == b"keep existing file"


def test_cooldown_invalid_crops_and_invalid_labels_are_rejected(tmp_path):
    recorder = make_recorder(tmp_path, cooldown_ms=200)
    assert not recorder.record("UNKNOWN", make_crop(), make_crop(), now=1.0).accepted
    assert not recorder.record(CaptureLabel.OPEN, None, make_crop(), now=1.0).accepted
    assert not recorder.record(CaptureLabel.OPEN, make_crop(8), make_crop(), now=1.0).accepted
    float_crop = EyeCrop(np.zeros((16, 16, 3), dtype=np.float32), BoundingBox(0, 0, 16, 16))
    assert not recorder.record(CaptureLabel.OPEN, float_crop, make_crop(), now=1.0).accepted

    assert recorder.record(CaptureLabel.CLOSED, make_crop(), make_crop(), now=1.0).accepted
    too_soon = recorder.record(CaptureLabel.CLOSED, make_crop(), make_crop(), now=1.1)
    assert not too_soon.accepted
    assert "Cooldown" in too_soon.message
    assert recorder.record(CaptureLabel.CLOSED, make_crop(), make_crop(), now=1.21).accepted


@pytest.mark.parametrize("field,value", [("subject_id", ""), ("session_id", "../private")])
def test_identifiers_must_be_nonblank_and_filename_safe(tmp_path, field, value):
    with pytest.raises(ValueError):
        make_recorder(tmp_path, **{field: value})
