from __future__ import annotations

import json

import numpy as np
import pytest

from changepoint_lab import (
    LatentStateResult,
    PELT,
    SegmentationResult,
)
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar
from changepoint_lab.core.datatypes import (
    ChangePointResult,
    ModelSelectionResult,
    OnlineProbabilityResult,
    PosteriorSampleResult,
)


def test_segmentation_result_round_trips_through_json() -> None:
    result = SegmentationResult(
        indices=np.array([3]),
        score=1.5,
        labels=np.array([0, 0, 0, 1]),
        costs_per_segment=np.array([0.25, 1.25]),
        method_name="example",
        objective_orientation="minimize",
        metadata={"source": "unit"},
    )

    payload = json.loads(json.dumps(result.to_dict()))
    restored = SegmentationResult.from_dict(payload)

    assert restored.indices.tolist() == [3]
    assert restored.labels is not None
    assert restored.labels.tolist() == [0, 0, 0, 1]
    assert restored.costs_per_segment is not None
    assert restored.costs_per_segment.tolist() == [0.25, 1.25]
    assert restored.metadata == {"source": "unit"}


def test_online_probability_result_round_trips_core_fields() -> None:
    result = OnlineProbabilityResult(
        indices=np.array([2]),
        cp_prob=np.array([0.1, 0.2, 0.7]),
        map_run_length=np.array([1, 2, 0]),
        pred_mean=np.array([0.5, 0.4, 0.3]),
        method_name="bocpd",
        boundary_convention="time_index",
    )

    restored = OnlineProbabilityResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored.indices.tolist() == [2]
    assert restored.cp_prob.tolist() == [0.1, 0.2, 0.7]
    assert restored.map_run_length.tolist() == [1, 2, 0]
    assert restored.pred_mean is not None
    assert restored.pred_mean.tolist() == [0.5, 0.4, 0.3]


def test_generic_result_round_trip_preserves_common_contract() -> None:
    result = ChangePointResult(indices=np.array([1, 4]), method_name="generic")

    restored = ChangePointResult.from_dict(result.to_dict())

    assert restored.indices.tolist() == [1, 4]
    assert restored.method_name == "generic"
    assert restored.boundary_convention == "right_exclusive"


@pytest.mark.parametrize(
    "indices",
    [
        np.array([3, 1]),
        np.array([1, 1]),
        np.array([-1]),
        np.array([[1]]),
        np.array([1.5]),
    ],
)
def test_changepoint_indices_fail_for_invalid_domain(indices: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ChangePointResult(indices=indices)


def test_result_arrays_are_copied_and_read_only() -> None:
    source = np.array([2], dtype=int)
    result = ChangePointResult(indices=source)

    source[0] = 9

    assert result.indices.tolist() == [2]
    assert not result.indices.flags.writeable
    with pytest.raises(ValueError):
        result.indices[0] = 3


def test_labels_and_cost_arrays_validate_shape_and_domain() -> None:
    with pytest.raises(ValueError, match="labels"):
        ChangePointResult(indices=np.array([2]), labels=np.array([[0, 1]]))
    with pytest.raises(ValueError, match="labels"):
        ChangePointResult(indices=np.array([2]), labels=np.array([0, -1]))
    with pytest.raises(ValueError, match="costs_per_segment"):
        SegmentationResult(indices=np.array([2]), costs_per_segment=np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="one value per segment"):
        SegmentationResult(indices=np.array([2]), costs_per_segment=np.array([1.0]))


def test_probability_and_latent_arrays_validate_dimensions() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        OnlineProbabilityResult(
            indices=np.array([1]),
            cp_prob=np.array([0.2, 1.2]),
            map_run_length=np.array([1, 0]),
        )
    with pytest.raises(ValueError, match="matching lengths"):
        OnlineProbabilityResult(
            indices=np.array([1]),
            cp_prob=np.array([0.2, 0.8]),
            map_run_length=np.array([1]),
        )
    with pytest.raises(ValueError, match="segment_durations"):
        LatentStateResult(
            indices=np.array([1]),
            states=np.array([0, 1]),
            segment_durations=np.array([1]),
        )


def test_deserialization_validates_schema_enums() -> None:
    payload = ChangePointResult(indices=np.array([1])).to_dict()
    payload["boundary_convention"] = "unknown"
    with pytest.raises(ValueError, match="boundary_convention"):
        ChangePointResult.from_dict(payload)

    payload = ChangePointResult(indices=np.array([1])).to_dict()
    payload["objective_orientation"] = "sideways"
    with pytest.raises(ValueError, match="objective_orientation"):
        ChangePointResult.from_dict(payload)

    payload = ChangePointResult(indices=np.array([1])).to_dict()
    payload["result_type"] = "OtherResult"
    with pytest.raises(ValueError, match="result_type"):
        ChangePointResult.from_dict(payload)


@pytest.mark.parametrize(
    "result",
    [
        ChangePointResult(indices=np.array([1]), method_name="generic"),
        SegmentationResult(
            indices=np.array([2]),
            labels=np.array([0, 0, 1]),
            costs_per_segment=np.array([0.5, 1.5]),
            objective_orientation="minimize",
        ),
        OnlineProbabilityResult(
            indices=np.array([1]),
            boundary_convention="time_index",
            cp_prob=np.array([0.1, 0.4, 0.2]),
            map_run_length=np.array([1, 2, 0]),
            pred_mean=np.array([0.0, 0.1, 0.2]),
        ),
        PosteriorSampleResult(
            indices=np.array([1, 3]),
            boundary_convention="periodic_bin_end",
            samples=((1,), (1, 3)),
            log_posteriors=np.array([-3.0, -2.0]),
            changepoint_hist=np.array([0, 2, 0, 1]),
        ),
        LatentStateResult(
            indices=np.array([2]),
            states=np.array([0, 0, 1]),
            segment_durations=np.array([0, 2, 1]),
        ),
        ModelSelectionResult(
            indices=np.array([2]),
            selected_model=2,
            criterion_values=np.array([4.0, 2.0, 3.0]),
            objective_orientation="minimize",
        ),
    ],
)
def test_all_stable_result_subclasses_round_trip_public_payloads(
    result: ChangePointResult,
) -> None:
    payload = json.loads(json.dumps(result.to_dict()))
    restored = type(result).from_dict(payload)

    assert restored.to_dict() == result.to_dict()


def test_latent_state_result_serializes_typed_state_fields() -> None:
    result = LatentStateResult(
        indices=np.array([2]),
        states=np.array([0, 0, 1, 1]),
        segment_durations=np.array([0, 2, 0, 2]),
        method_name="hsmm",
    )

    payload = result.to_dict(include_metadata=False)

    assert payload["states"] == [0, 0, 1, 1]
    assert payload["segment_durations"] == [0, 2, 0, 2]


def test_predict_before_fit_raises_domain_message() -> None:
    detector = PELT(cost_fn=NormalMeanKnownVar(sigma2=1.0), penalty=1.0)

    with pytest.raises(RuntimeError, match="Call fit before predict"):
        detector.predict()


def test_detector_input_validation_uses_domain_message() -> None:
    detector = PELT(cost_fn=NormalMeanKnownVar(sigma2=1.0), penalty=1.0)

    with pytest.raises(TypeError, match="`x` must be np.ndarray"):
        detector.fit([1.0, 2.0])  # type: ignore[arg-type]
