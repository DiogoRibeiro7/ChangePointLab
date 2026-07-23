from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, Optional, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray


BoundaryConvention: TypeAlias = Literal[
    "right_exclusive",
    "time_index",
    "periodic_bin_end",
]
ObjectiveOrientation: TypeAlias = Literal["minimize", "maximize"]
ArrayI: TypeAlias = NDArray[np.int_]
ArrayF: TypeAlias = NDArray[np.float64]


def _array(values: Sequence[Any] | np.ndarray, *, dtype: type) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_jsonable(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class ChangePointResult:
    """Container for detected change points.

    Parameters
    ----------
    indices : np.ndarray
        Sorted 1D array of changepoint indices.
    score : float, optional
        Optional overall score or objective value.
    metadata : Mapping[str, Any], optional
        Additional algorithm-specific outputs.
    """

    indices: ArrayI
    score: Optional[float] = None
    labels: ArrayI | None = None
    method_name: str | None = None
    boundary_convention: BoundaryConvention = "right_exclusive"
    objective_orientation: ObjectiveOrientation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _array(self.indices, dtype=int))
        if self.labels is not None:
            object.__setattr__(self, "labels", _array(self.labels, dtype=int))

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible representation of the result."""
        payload: dict[str, Any] = {
            "result_type": type(self).__name__,
            "indices": self.indices.tolist(),
            "score": self.score,
            "labels": None if self.labels is None else self.labels.tolist(),
            "method_name": self.method_name,
            "boundary_convention": self.boundary_convention,
            "objective_orientation": self.objective_orientation,
            "provenance": _to_jsonable(self.provenance),
        }
        if include_metadata:
            payload["metadata"] = _to_jsonable(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChangePointResult:
        """Rebuild a generic changepoint result from :meth:`to_dict` output."""
        labels = payload.get("labels")
        return cls(
            indices=_array(payload["indices"], dtype=int),
            score=payload.get("score"),
            labels=None if labels is None else _array(labels, dtype=int),
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "right_exclusive"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
        )


@dataclass(frozen=True)
class SegmentationResult(ChangePointResult):
    """Offline segmentation result with segment labels and segment costs."""

    costs_per_segment: ArrayF | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.costs_per_segment is not None:
            object.__setattr__(
                self,
                "costs_per_segment",
                _array(self.costs_per_segment, dtype=float),
            )

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["costs_per_segment"] = (
            None if self.costs_per_segment is None else self.costs_per_segment.tolist()
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SegmentationResult:
        labels = payload.get("labels")
        costs = payload.get("costs_per_segment")
        return cls(
            indices=_array(payload["indices"], dtype=int),
            score=payload.get("score"),
            labels=None if labels is None else _array(labels, dtype=int),
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "right_exclusive"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
            costs_per_segment=None if costs is None else _array(costs, dtype=float),
        )


@dataclass(frozen=True)
class OnlineProbabilityResult(ChangePointResult):
    """Online result with changepoint probabilities and run-length summaries."""

    cp_prob: ArrayF = field(default_factory=lambda: np.array([], dtype=float))
    map_run_length: ArrayI = field(default_factory=lambda: np.array([], dtype=int))
    pred_mean: ArrayF | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "cp_prob", _array(self.cp_prob, dtype=float))
        object.__setattr__(self, "map_run_length", _array(self.map_run_length, dtype=int))
        if self.pred_mean is not None:
            object.__setattr__(self, "pred_mean", _array(self.pred_mean, dtype=float))

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["cp_prob"] = self.cp_prob.tolist()
        payload["map_run_length"] = self.map_run_length.tolist()
        payload["pred_mean"] = None if self.pred_mean is None else self.pred_mean.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OnlineProbabilityResult:
        return cls(
            indices=_array(payload["indices"], dtype=int),
            score=payload.get("score"),
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "time_index"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
            cp_prob=_array(payload.get("cp_prob", []), dtype=float),
            map_run_length=_array(payload.get("map_run_length", []), dtype=int),
            pred_mean=None
            if payload.get("pred_mean") is None
            else _array(payload["pred_mean"], dtype=float),
        )


@dataclass(frozen=True)
class PosteriorSampleResult(ChangePointResult):
    """Posterior-sampling result with sampled changepoint sets and log posterior trace."""

    samples: tuple[tuple[int, ...], ...] = ()
    log_posteriors: ArrayF = field(default_factory=lambda: np.array([], dtype=float))
    changepoint_hist: ArrayI | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "samples",
            tuple(tuple(int(item) for item in sample) for sample in self.samples),
        )
        object.__setattr__(self, "log_posteriors", _array(self.log_posteriors, dtype=float))
        if self.changepoint_hist is not None:
            object.__setattr__(self, "changepoint_hist", _array(self.changepoint_hist, dtype=int))

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["samples"] = [list(sample) for sample in self.samples]
        payload["log_posteriors"] = self.log_posteriors.tolist()
        payload["changepoint_hist"] = (
            None if self.changepoint_hist is None else self.changepoint_hist.tolist()
        )
        return payload


@dataclass(frozen=True)
class LatentStateResult(ChangePointResult):
    """Latent-state decoding result with state sequence and segment end durations."""

    states: ArrayI = field(default_factory=lambda: np.array([], dtype=int))
    segment_durations: ArrayI | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "states", _array(self.states, dtype=int))
        if self.segment_durations is not None:
            object.__setattr__(
                self,
                "segment_durations",
                _array(self.segment_durations, dtype=int),
            )

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["states"] = self.states.tolist()
        payload["segment_durations"] = (
            None if self.segment_durations is None else self.segment_durations.tolist()
        )
        return payload


@dataclass(frozen=True)
class ModelSelectionResult(ChangePointResult):
    """Model-selection result with criterion values over candidate segment counts."""

    selected_model: int | None = None
    criterion_values: ArrayF | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.criterion_values is not None:
            object.__setattr__(
                self,
                "criterion_values",
                _array(self.criterion_values, dtype=float),
            )


DetectorT = TypeVar("DetectorT", covariant=True)


@runtime_checkable
class OfflineDetector(Protocol[DetectorT]):
    """Protocol for offline estimators that fit and then predict changepoints."""

    def fit(self, x: np.ndarray) -> DetectorT:
        """Fit the detector to a complete array."""
        ...

    def predict(self, x: np.ndarray | None = None) -> ChangePointResult:
        """Return changepoints for fitted data or a supplied array."""
        ...

    def fit_predict(self, x: np.ndarray) -> ChangePointResult:
        """Fit and predict in one call."""
        ...

    def get_params(self) -> Mapping[str, Any]:
        """Return constructor parameters."""
        ...


@runtime_checkable
class OnlineDetector(Protocol):
    """Protocol for online/probability-trace detectors."""

    def run(self, x: np.ndarray) -> Any:
        """Process a sequence and return online probabilities."""
        ...


@runtime_checkable
class LatentStateDecoder(Protocol):
    """Protocol for models that decode latent state sequences."""

    def decode_viterbi(self, loglik_tk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode a Viterbi state path and segment-duration indicators."""
        ...


@runtime_checkable
class PosteriorSampler(Protocol):
    """Protocol for models whose fitted state exposes posterior samples."""

    @property
    def result(self) -> Any:
        """Return posterior sample summaries after fitting."""
        ...
