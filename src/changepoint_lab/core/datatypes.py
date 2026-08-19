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
ArrayI: TypeAlias = NDArray[np.integer[Any]]
ArrayF: TypeAlias = NDArray[np.floating[Any]]

_BOUNDARY_CONVENTIONS = {"right_exclusive", "time_index", "periodic_bin_end"}
_OBJECTIVE_ORIENTATIONS = {"minimize", "maximize"}


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _integer_array(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    non_negative: bool = False,
    sorted_unique: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1-D integer array.")
    if raw.size and raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integers.")
    array = np.array(values, dtype=int, copy=True)
    if non_negative and np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    if sorted_unique and array.size > 1 and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be sorted and duplicate-free.")
    return _readonly(array)


def _float_array(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    probability: bool = False,
) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D numeric array.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if probability and np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return _readonly(array)


def _validate_common_fields(result: ChangePointResult) -> None:
    if result.score is not None:
        score = float(result.score)
        if not np.isfinite(score):
            raise ValueError("score must be finite when provided.")
        object.__setattr__(result, "score", score)
    if result.boundary_convention not in _BOUNDARY_CONVENTIONS:
        raise ValueError("boundary_convention must be a known convention.")
    if (
        result.objective_orientation is not None
        and result.objective_orientation not in _OBJECTIVE_ORIENTATIONS
    ):
        raise ValueError("objective_orientation must be 'minimize', 'maximize', or None.")


def _validate_result_type(payload: Mapping[str, Any], expected: str) -> None:
    result_type = payload.get("result_type")
    if result_type is not None and result_type != expected:
        raise ValueError(f"payload result_type must be {expected}.")


def _validate_equal_lengths(arrays: Mapping[str, np.ndarray | None]) -> None:
    lengths = {name: array.size for name, array in arrays.items() if array is not None}
    non_empty = {name: length for name, length in lengths.items() if length > 0}
    if len(set(non_empty.values())) > 1:
        joined = ", ".join(non_empty)
        raise ValueError(f"{joined} must have matching lengths.")


def _validate_changepoint_samples(samples: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    validated: list[tuple[int, ...]] = []
    for index, sample in enumerate(samples):
        array = _integer_array(
            sample,
            name=f"samples[{index}]",
            non_negative=True,
            sorted_unique=True,
        )
        validated.append(tuple(int(item) for item in array))
    return tuple(validated)


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
        object.__setattr__(
            self,
            "indices",
            _integer_array(
                self.indices,
                name="indices",
                non_negative=True,
                sorted_unique=True,
            ),
        )
        if self.labels is not None:
            object.__setattr__(
                self,
                "labels",
                _integer_array(self.labels, name="labels", non_negative=True),
            )
        _validate_common_fields(self)

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
        _validate_result_type(payload, cls.__name__)
        labels = payload.get("labels")
        return cls(
            indices=payload["indices"],
            score=payload.get("score"),
            labels=labels,
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
                _float_array(self.costs_per_segment, name="costs_per_segment"),
            )
            if self.costs_per_segment.size != self.indices.size + 1:
                raise ValueError("costs_per_segment must have one value per segment.")

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["costs_per_segment"] = (
            None if self.costs_per_segment is None else self.costs_per_segment.tolist()
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SegmentationResult:
        _validate_result_type(payload, cls.__name__)
        labels = payload.get("labels")
        costs = payload.get("costs_per_segment")
        return cls(
            indices=payload["indices"],
            score=payload.get("score"),
            labels=labels,
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "right_exclusive"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
            costs_per_segment=costs,
        )


@dataclass(frozen=True)
class OnlineProbabilityResult(ChangePointResult):
    """Online result with changepoint probabilities and run-length summaries."""

    cp_prob: ArrayF = field(default_factory=lambda: np.array([], dtype=float))
    map_run_length: ArrayI = field(default_factory=lambda: np.array([], dtype=int))
    pred_mean: ArrayF | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "cp_prob",
            _float_array(self.cp_prob, name="cp_prob", probability=True),
        )
        object.__setattr__(
            self,
            "map_run_length",
            _integer_array(self.map_run_length, name="map_run_length", non_negative=True),
        )
        if self.pred_mean is not None:
            object.__setattr__(
                self,
                "pred_mean",
                _float_array(self.pred_mean, name="pred_mean"),
            )
        _validate_equal_lengths(
            {
                "cp_prob": self.cp_prob,
                "map_run_length": self.map_run_length,
                "pred_mean": self.pred_mean,
            }
        )

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["cp_prob"] = self.cp_prob.tolist()
        payload["map_run_length"] = self.map_run_length.tolist()
        payload["pred_mean"] = None if self.pred_mean is None else self.pred_mean.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OnlineProbabilityResult:
        _validate_result_type(payload, cls.__name__)
        return cls(
            indices=payload["indices"],
            score=payload.get("score"),
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "time_index"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
            cp_prob=payload.get("cp_prob", []),
            map_run_length=payload.get("map_run_length", []),
            pred_mean=payload.get("pred_mean"),
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
            _validate_changepoint_samples(self.samples),
        )
        object.__setattr__(
            self,
            "log_posteriors",
            _float_array(self.log_posteriors, name="log_posteriors"),
        )
        if self.changepoint_hist is not None:
            object.__setattr__(
                self,
                "changepoint_hist",
                _integer_array(self.changepoint_hist, name="changepoint_hist", non_negative=True),
            )

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["samples"] = [list(sample) for sample in self.samples]
        payload["log_posteriors"] = self.log_posteriors.tolist()
        payload["changepoint_hist"] = (
            None if self.changepoint_hist is None else self.changepoint_hist.tolist()
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PosteriorSampleResult:
        """Rebuild a posterior-sampling result from :meth:`to_dict` output."""
        _validate_result_type(payload, cls.__name__)
        return cls(
            indices=payload["indices"],
            score=payload.get("score"),
            labels=payload.get("labels"),
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "periodic_bin_end"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
            samples=tuple(tuple(sample) for sample in payload.get("samples", ())),
            log_posteriors=payload.get("log_posteriors", []),
            changepoint_hist=payload.get("changepoint_hist"),
        )


@dataclass(frozen=True)
class LatentStateResult(ChangePointResult):
    """Latent-state decoding result with state sequence and segment end durations."""

    states: ArrayI = field(default_factory=lambda: np.array([], dtype=int))
    segment_durations: ArrayI | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "states",
            _integer_array(self.states, name="states", non_negative=True),
        )
        if self.segment_durations is not None:
            object.__setattr__(
                self,
                "segment_durations",
                _integer_array(
                    self.segment_durations,
                    name="segment_durations",
                    non_negative=True,
                ),
            )
            if self.states.size and self.segment_durations.size != self.states.size:
                raise ValueError("segment_durations must match states length.")

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["states"] = self.states.tolist()
        payload["segment_durations"] = (
            None if self.segment_durations is None else self.segment_durations.tolist()
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LatentStateResult:
        """Rebuild a latent-state result from :meth:`to_dict` output."""
        _validate_result_type(payload, cls.__name__)
        return cls(
            indices=payload["indices"],
            score=payload.get("score"),
            labels=payload.get("labels"),
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "right_exclusive"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
            states=payload.get("states", []),
            segment_durations=payload.get("segment_durations"),
        )


@dataclass(frozen=True)
class ModelSelectionResult(ChangePointResult):
    """Model-selection result with criterion values over candidate segment counts."""

    selected_model: int | None = None
    criterion_values: ArrayF | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.selected_model is not None and self.selected_model < 0:
            raise ValueError("selected_model must be non-negative when provided.")
        if self.criterion_values is not None:
            object.__setattr__(
                self,
                "criterion_values",
                _float_array(self.criterion_values, name="criterion_values"),
            )

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_metadata=include_metadata)
        payload["selected_model"] = self.selected_model
        payload["criterion_values"] = (
            None if self.criterion_values is None else self.criterion_values.tolist()
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelSelectionResult:
        """Rebuild a model-selection result from :meth:`to_dict` output."""
        _validate_result_type(payload, cls.__name__)
        return cls(
            indices=payload["indices"],
            score=payload.get("score"),
            labels=payload.get("labels"),
            method_name=payload.get("method_name"),
            boundary_convention=payload.get("boundary_convention", "right_exclusive"),
            objective_orientation=payload.get("objective_orientation"),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
            selected_model=payload.get("selected_model"),
            criterion_values=payload.get("criterion_values"),
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
