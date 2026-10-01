from src.utils import FPSCounter


def test_fps_counter_uses_monotonic_interval():
    fps = FPSCounter(window_size=3)

    assert fps.update(1.0) == 0.0
    assert fps.update(1.5) == 2.0
    assert fps.update(2.0) == 2.0
    assert fps.update(2.5) == 2.0


def test_fps_counter_ignores_non_increasing_timestamp():
    fps = FPSCounter()
    fps.update(1.0)
    fps.update(2.0)

    assert fps.update(2.0) == 1.0
