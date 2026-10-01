import numpy as np

from src.eye_cropper import EyeCropper, Point


def test_bounding_box_applies_padding_and_clips_to_frame():
    cropper = EyeCropper(output_size=None, padding_ratio=0.5)
    points = [Point(1, 2), Point(5, 6)]

    box = cropper.bounding_box(points, (10, 10, 3))

    assert box is not None
    assert (box.x, box.y, box.width, box.height) == (0, 0, 8, 9)


def test_crop_returns_resized_image_and_source_box():
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    cropper = EyeCropper(output_size=16, padding_ratio=0)

    crop = cropper.crop(frame, [Point(10, 10), Point(20, 18)])

    assert crop is not None
    assert crop.image.shape == (16, 16, 3)
    assert crop.box.x == 10
    assert crop.box.y == 10


def test_crop_rejects_empty_or_outside_regions():
    cropper = EyeCropper(output_size=16, padding_ratio=0)

    assert cropper.crop(np.zeros((10, 10, 3), dtype=np.uint8), []) is None
    assert cropper.crop(np.zeros((10, 10, 3), dtype=np.uint8), [Point(20, 20)]) is None
