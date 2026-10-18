from mlflow.entities.assessment import (
    AssessmentError,
    Expectation,
    ExpectationValue,
    Feedback,
    FeedbackValue,
)
import pytest

from smoke_test.support import assert_native_value, error

pytestmark = pytest.mark.family("assessments")


def check(smoke, tid, assessment, field, value):
    loaded = smoke.rest.get_assessment(tid, assessment.assessment_id)
    assert_native_value(loaded.value, value)
    document = smoke.document(
        "assessments", {"trace_id": tid, "assessment_id": assessment.assessment_id}
    )
    assert_native_value(document["content"][field]["value"], value)
    joined = smoke.rest.get_trace_info(tid).assessments
    assert assessment.assessment_id in {item.assessment_id for item in joined}
    return loaded, document


@pytest.mark.parametrize("value", [True, 7, 0.5, "$yes", [1, "a"], {"answer": [1, True]}])
def test_feedback_native_values(smoke, value):
    trace = smoke.trace()
    tid = trace.info.trace_id
    created = smoke.rest.create_assessment(Feedback(name="quality", value=value, trace_id=tid))
    loaded, document = check(smoke, tid, created, "feedback", value)
    assert loaded.valid is True
    assert document["experiment_id"] == trace.info.experiment_id


@pytest.mark.parametrize("value", [[1, {"answer": True}], {"answer": [1, "yes"]}])
def test_expectation_update(smoke, value):
    tid = smoke.trace().info.trace_id
    created = smoke.rest.create_assessment(Expectation(name="expected", value="old", trace_id=tid))
    check(smoke, tid, created, "expectation", "old")
    updated = smoke.rest.update_assessment(
        tid, created.assessment_id, expectation=ExpectationValue(value)
    )
    check(smoke, tid, updated, "expectation", value)


def test_feedback_updates_metadata_and_identity(smoke):
    tid = smoke.trace().info.trace_id
    created = smoke.rest.create_assessment(
        Feedback(
            name="quality",
            value=True,
            trace_id=tid,
            rationale="old",
            metadata={"keep": "yes", "replace": "old"},
        )
    )
    check(smoke, tid, created, "feedback", True)
    updated = smoke.rest.update_assessment(
        tid,
        created.assessment_id,
        name="renamed",
        rationale="new",
        metadata={"replace": "$new", "extra": "value"},
        feedback=FeedbackValue(False),
    )
    loaded, document = check(smoke, tid, updated, "feedback", False)
    assert loaded.name == document["content"]["assessment_name"] == "renamed"
    assert loaded.rationale == document["content"]["rationale"] == "new"
    assert loaded.metadata == {"keep": "yes", "replace": "$new", "extra": "value"}
    assert loaded.source == created.source and loaded.create_time_ms == created.create_time_ms
    assert loaded.last_update_time_ms >= created.last_update_time_ms


def test_feedback_error_and_recovery(smoke):
    tid = smoke.trace().info.trace_id
    created = smoke.rest.create_assessment(Feedback(name="quality", value=True, trace_id=tid))
    check(smoke, tid, created, "feedback", True)
    failure = AssessmentError(error_code="JUDGE_ERROR", error_message="smoke judge failed")
    smoke.rest.update_assessment(tid, created.assessment_id, feedback=FeedbackValue(None, failure))
    loaded = smoke.rest.get_assessment(tid, created.assessment_id)
    assert loaded.feedback.error == failure
    assert (
        smoke.document("assessments", {"assessment_id": created.assessment_id})["content"][
            "feedback"
        ]["error"]["error_code"]
        == "JUDGE_ERROR"
    )
    recovered = smoke.rest.update_assessment(
        tid, created.assessment_id, feedback=FeedbackValue(True)
    )
    loaded, document = check(smoke, tid, recovered, "feedback", True)
    assert loaded.feedback.error is None
    assert not document["content"]["feedback"].get("error")


def test_override_delete_restores_original(smoke):
    tid = smoke.trace().info.trace_id
    original = smoke.rest.create_assessment(Feedback(name="quality", value=False, trace_id=tid))
    check(smoke, tid, original, "feedback", False)
    override = smoke.rest.create_assessment(
        Feedback(name="quality", value=True, trace_id=tid, overrides=original.assessment_id)
    )
    check(smoke, tid, override, "feedback", True)
    assert smoke.rest.get_assessment(tid, original.assessment_id).valid is False
    assert (
        smoke.document("assessments", {"assessment_id": original.assessment_id})["content"]["valid"]
        is False
    )
    smoke.rest.delete_assessment(tid, override.assessment_id)
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.rest.get_assessment(tid, override.assessment_id)
    assert smoke.documents("assessments", {"assessment_id": override.assessment_id}) == []
    assert smoke.rest.get_assessment(tid, original.assessment_id).valid is True
    assert (
        smoke.document("assessments", {"assessment_id": original.assessment_id})["content"]["valid"]
        is True
    )
    smoke.rest.delete_assessment(tid, override.assessment_id)
    assert smoke.documents("assessments", {"assessment_id": override.assessment_id}) == []
    assert len(smoke.rest.get_trace_info(tid).assessments) == 1


def test_assessment_scoping_and_invalid_update(smoke):
    eid = smoke.experiment()
    first, second = smoke.trace(eid, "first"), smoke.trace(eid, "second")
    tid = first.info.trace_id
    created = smoke.rest.create_assessment(Feedback(name="quality", value=True, trace_id=tid))
    check(smoke, tid, created, "feedback", True)
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.rest.get_assessment(second.info.trace_id, created.assessment_id)
    with error("INVALID_PARAMETER_VALUE"):
        smoke.rest.update_assessment(
            tid, created.assessment_id, expectation=ExpectationValue("invalid")
        )
    check(smoke, tid, created, "feedback", True)
    assert smoke.rest.get_trace_info(second.info.trace_id).assessments == []
    assert smoke.documents("assessments", {"trace_id": second.info.trace_id}) == []
