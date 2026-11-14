"""Contracts for named offline detector factories."""

import numpy as np
import pytest

from changepoint_lab import ChangePointResult, DetectorRegistry


class StubDetector:
    """Stateful test detector with a known changepoint."""

    def __init__(self, changepoint: int) -> None:
        self.changepoint = changepoint
        self.calls = 0

    def fit_predict(self, data: np.ndarray) -> ChangePointResult:
        self.calls += 1
        return ChangePointResult(indices=np.asarray([self.changepoint]))


def test_registry_returns_new_instances_and_preserves_order() -> None:
    registry = DetectorRegistry()
    registry.register("first", lambda: StubDetector(2))
    registry.register("second", lambda: StubDetector(3))
    assert registry.names == ("first", "second")
    first = registry.build()
    second = registry.build()
    assert list(first) == ["first", "second"]
    assert first["first"] is not second["first"]
    assert list(registry.build(["second", "first"])) == ["second", "first"]


def test_compare_creates_fresh_detector_each_time() -> None:
    registry = DetectorRegistry()
    constructed: list[StubDetector] = []

    def factory() -> StubDetector:
        detector = StubDetector(2)
        constructed.append(detector)
        return detector

    registry.register("stub", factory)
    for _ in range(2):
        (result,) = registry.compare([0.0, 0.0, 1.0, 1.0], actual=[2])
        assert result.scores is not None and result.scores.f1 == 1.0
    assert len(constructed) == 2
    assert constructed[0] is not constructed[1]
    assert [detector.calls for detector in constructed] == [1, 1]


def test_registries_are_independent() -> None:
    left = DetectorRegistry()
    right = DetectorRegistry()
    left.register("only-left", lambda: StubDetector(1))
    assert right.names == ()


def test_invalid_registration_and_selection() -> None:
    registry = DetectorRegistry()
    for name in ("", "  ", " padded "):
        with pytest.raises(ValueError):
            registry.register(name, lambda: StubDetector(1))
    with pytest.raises(TypeError):
        registry.register("bad", None)
    registry.register("valid", lambda: StubDetector(1))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("valid", lambda: StubDetector(2))
    with pytest.raises(KeyError, match="missing"):
        registry.build(["missing"])
    with pytest.raises(ValueError, match="unique"):
        registry.build(["valid", "valid"])
    with pytest.raises(ValueError, match="no detectors"):
        registry.build([])


def test_factory_failures_are_named() -> None:
    registry = DetectorRegistry()

    def fail() -> StubDetector:
        raise RuntimeError("construction failed")

    registry.register("broken", fail)
    with pytest.raises(RuntimeError, match="broken"):
        registry.build()

    other = DetectorRegistry()
    other.register("incompatible", lambda: object())
    with pytest.raises(TypeError, match="incompatible"):
        other.build()
