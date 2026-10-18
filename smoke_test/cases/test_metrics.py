from importlib.util import find_spec
import json

from mlflow.entities import TraceState
from mlflow.entities.assessment import Feedback
from mlflow.tracing import constant
import pytest

from smoke_test.support import assert_native_value, error, span

AVAILABLE = find_spec("mlflow.entities.trace_metrics") is not None
pytestmark = [
    pytest.mark.family("metrics"),
    pytest.mark.upstream("Trace metric HTTP API is available starting at MLflow 3.9"),
    pytest.mark.skipif(not AVAILABLE, reason="mlflow.entities.trace_metrics absent upstream"),
]


def query(smoke, eid, view, metric, aggregation="COUNT", **kwargs):
    from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType

    return smoke.rest.query_trace_metrics(
        [eid],
        MetricViewType(view),
        metric,
        [MetricAggregation(AggregationType(aggregation))],
        **kwargs,
    )


def test_trace_counts_status_dimension_and_scope(smoke):
    eid = smoke.experiment()
    for state in (TraceState.OK, TraceState.OK, TraceState.ERROR):
        smoke.trace_info(eid, state=state)
    result = query(smoke, eid, "TRACES", "trace_count", dimensions=["trace_status"])
    assert {point.dimensions["trace_status"]: point.values for point in result} == {
        "OK": {"COUNT": 2},
        "ERROR": {"COUNT": 1},
    }
    assert len(smoke.documents("traces", {"experiment_id": eid})) == 3
    other = smoke.experiment("other")
    assert query(smoke, other, "TRACES", "trace_count")[0].values == {"COUNT": 0}


@pytest.mark.parametrize(
    "filters,expected",
    [
        (["trace.status = 'OK'"], 2),
        (["trace.tag.`team.name` = '$risk'"], 2),
        (["trace.metadata.`session.key` = 'one'"], 1),
        (["trace.status = 'OK'", "trace.tag.`team.name` = '$risk'"], 1),
        (["trace.status = 'IN_PROGRESS'"], 0),
    ],
)
def test_trace_metric_filters(smoke, filters, expected):
    eid = smoke.experiment()
    for state, team, session in [
        (TraceState.OK, "$risk", "one"),
        (TraceState.ERROR, "$risk", "two"),
        (TraceState.OK, "other", "two"),
    ]:
        smoke.trace_info(
            eid, state=state, tags={"team.name": team}, trace_metadata={"session.key": session}
        )
    result = query(smoke, eid, "TRACES", "trace_count", filters=filters)
    assert len(result) == 1 and result[0].values == {"COUNT": expected}
    assert len(smoke.documents("traces", {"experiment_id": eid})) == 3


def test_latency_time_buckets_and_limit(smoke):
    eid = smoke.experiment()
    for timestamp, duration in [(0, 10), (1000, 30), (60000, 50)]:
        smoke.trace_info(
            eid,
            request_time=timestamp,
            execution_duration=duration,
            tags={"mlflow.traceName": "agent"},
        )
    found = query(
        smoke,
        eid,
        "TRACES",
        "latency",
        "AVG",
        dimensions=["trace_name"],
        time_interval_seconds=60,
        start_time_ms=0,
        end_time_ms=60000,
    )
    assert [(point.dimensions, point.values) for point in found] == [
        ({"time_bucket": "1970-01-01T00:00:00+00:00", "trace_name": "agent"}, {"AVG": 20}),
        ({"time_bucket": "1970-01-01T00:01:00+00:00", "trace_name": "agent"}, {"AVG": 50}),
    ]
    assert sorted(
        item["execution_duration"] for item in smoke.documents("traces", {"experiment_id": eid})
    ) == [10, 30, 50]
    limited = query(
        smoke,
        eid,
        "TRACES",
        "latency",
        "AVG",
        dimensions=["trace_name"],
        time_interval_seconds=60,
        start_time_ms=0,
        end_time_ms=60000,
        max_results=1,
    )
    assert len(limited) == 1


def test_latency_percentile(smoke):
    from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType

    eid = smoke.experiment()
    for _ in range(3):
        smoke.trace_info(eid, execution_duration=10)
    aggregation = MetricAggregation(AggregationType.PERCENTILE, 95.0)
    result = smoke.rest.query_trace_metrics([eid], MetricViewType.TRACES, "latency", [aggregation])
    assert len(result) == 1 and result[0].values == {str(aggregation): 10}
    assert len(smoke.documents("traces", {"experiment_id": eid})) == 3


def test_token_sum(smoke):
    eid = smoke.experiment()
    for count in (10, 20):
        smoke.trace_info(
            eid,
            trace_metadata={
                constant.TraceMetadataKey.TOKEN_USAGE: json.dumps({"input_tokens": count})
            },
        )
    smoke.trace_info(eid)
    result = query(smoke, eid, "TRACES", "input_tokens", "SUM")
    assert len(result) == 1 and result[0].values == {"SUM": 30}
    docs = smoke.documents("traces", {"experiment_id": eid})
    assert sum((item.get("token_usage") or {}).get("input_tokens", 0) for item in docs) == 30


@pytest.mark.upstream("Session count metric absent in older trace metric APIs")
@pytest.mark.skipif(
    not hasattr(getattr(constant, "TraceMetricKey", None), "SESSION_COUNT"),
    reason="No session_count upstream",
)
def test_session_count(smoke):
    eid = smoke.experiment()
    for session in ("one", "one", "two"):
        smoke.trace_info(eid, trace_metadata={constant.TraceMetadataKey.TRACE_SESSION: session})
    result = query(smoke, eid, "TRACES", "session_count")
    assert result[0].values == {"COUNT": 2}
    assert len(smoke.documents("traces", {"experiment_id": eid})) == 3


def test_span_counts_and_nanosecond_buckets(smoke):
    eid = smoke.experiment()
    info = smoke.trace_info(eid, request_time=0)
    for index, start_time in enumerate((1000000, 59999999999, 60000000000, 61000000000), start=1):
        smoke.rest.log_spans(
            eid,
            [
                span(
                    info.trace_id,
                    index,
                    parent=1 if index > 1 else None,
                    start=start_time,
                    duration=1000000,
                )
            ],
        )
        assert smoke.rest.get_trace_info(info.trace_id).trace_id == info.trace_id
        assert len(smoke.documents("spans", {"trace_id": info.trace_id})) == index
    found = query(
        smoke,
        eid,
        "SPANS",
        "span_count",
        time_interval_seconds=60,
        start_time_ms=0,
        end_time_ms=120000,
    )
    assert [(point.dimensions, point.values) for point in found] == [
        ({"time_bucket": "1970-01-01T00:00:00+00:00"}, {"COUNT": 2}),
        ({"time_bucket": "1970-01-01T00:01:00+00:00"}, {"COUNT": 2}),
    ]


@pytest.mark.parametrize("value", [True, 7.0, "good", {"answer": 1}])
def test_assessment_count_groups_native_values(smoke, value):
    eid = smoke.experiment()
    info = smoke.trace_info(eid)
    created = smoke.rest.create_assessment(
        Feedback(name="quality", value=value, trace_id=info.trace_id)
    )
    assert smoke.rest.get_assessment(info.trace_id, created.assessment_id).value == value
    assert (
        smoke.document("assessments", {"assessment_id": created.assessment_id})["content"][
            "feedback"
        ]["value"]
        == value
    )
    result = query(smoke, eid, "ASSESSMENTS", "assessment_count", dimensions=["assessment_name"])
    assert [(point.dimensions, point.values) for point in result] == [
        ({"assessment_name": "quality"}, {"COUNT": 1})
    ]


def test_assessment_time_buckets(smoke):
    eid = smoke.experiment()
    info = smoke.trace_info(eid, request_time=0)
    for timestamp in (1, 59999, 60000, 61000):
        created = smoke.rest.create_assessment(
            Feedback(
                name="quality",
                value=True,
                trace_id=info.trace_id,
                create_time_ms=timestamp,
                last_update_time_ms=120000,
            )
        )
        assert (
            smoke.rest.get_assessment(info.trace_id, created.assessment_id).create_time_ms
            == timestamp
        )
        smoke.document("assessments", {"assessment_id": created.assessment_id})
    result = query(
        smoke,
        eid,
        "ASSESSMENTS",
        "assessment_count",
        time_interval_seconds=60,
        start_time_ms=0,
        end_time_ms=120000,
    )
    assert [(point.dimensions, point.values) for point in result] == [
        ({"time_bucket": "1970-01-01T00:00:00+00:00"}, {"COUNT": 2}),
        ({"time_bucket": "1970-01-01T00:01:00+00:00"}, {"COUNT": 2}),
    ]


def test_view_mismatch_rejected(smoke):
    eid = smoke.experiment()
    with error("INVALID_PARAMETER_VALUE"):
        query(smoke, eid, "TRACES", "trace_count", filters=["span.name = 'root'"])


@pytest.mark.upstream("Cost metrics absent upstream")
@pytest.mark.skipif(
    not hasattr(getattr(constant, "SpanMetricKey", None), "TOTAL_COST"),
    reason="No span cost metric upstream",
)
def test_span_cost_sum(smoke):
    eid = smoke.experiment()
    info = smoke.trace_info(eid)
    cost_key = constant.SpanAttributeKey.LLM_COST
    smoke.rest.log_spans(
        eid,
        [
            span(
                info.trace_id,
                attributes={
                    cost_key: json.dumps({"input_cost": 0.1, "output_cost": 0.2, "total_cost": 0.3})
                },
            )
        ],
    )
    assert smoke.rest.get_trace_info(info.trace_id).trace_id == info.trace_id
    assert len(smoke.documents("spans", {"trace_id": info.trace_id})) == 1
    result = query(smoke, eid, "SPANS", "total_cost", "SUM")
    assert result[0].values["SUM"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "dimension,attribute",
    [
        pytest.param(
            "span_model_name",
            "mlflow.llm.model",
            marks=[]
            if hasattr(getattr(constant, "SpanMetricDimensionKey", None), "SPAN_MODEL_NAME")
            else [pytest.mark.skip(reason="No model-name metric dimension upstream")],
        ),
        pytest.param(
            "span_model_provider",
            "mlflow.llm.provider",
            marks=[]
            if hasattr(getattr(constant, "SpanMetricDimensionKey", None), "SPAN_MODEL_PROVIDER")
            else [pytest.mark.skip(reason="No provider metric dimension upstream")],
        ),
    ],
)
def test_span_model_dimensions(smoke, dimension, attribute):
    eid = smoke.experiment()
    info = smoke.trace_info(eid)
    for number, value in enumerate(("first", "second"), start=1):
        smoke.rest.log_spans(
            eid,
            [
                span(
                    info.trace_id,
                    number,
                    parent=1 if number > 1 else None,
                    attributes={attribute: value},
                )
            ],
        )
        assert smoke.rest.get_trace_info(info.trace_id).trace_id == info.trace_id
        assert len(smoke.documents("spans", {"trace_id": info.trace_id})) == number
    result = query(smoke, eid, "SPANS", "span_count", dimensions=[dimension])
    assert {point.dimensions[dimension]: point.values for point in result} == {
        "first": {"COUNT": 1},
        "second": {"COUNT": 1},
    }


@pytest.mark.parametrize("duration,expected", [(999999, 0), (1500000, 1), (3250000, 3)])
def test_span_latency_milliseconds(smoke, duration, expected):
    eid = smoke.experiment()
    info = smoke.trace_info(eid)
    smoke.rest.log_spans(eid, [span(info.trace_id, duration=duration)])
    assert smoke.rest.get_trace_info(info.trace_id).trace_id == info.trace_id
    assert len(smoke.documents("spans", {"trace_id": info.trace_id})) == 1
    result = query(smoke, eid, "SPANS", "latency", "AVG")
    assert result[0].values == {"AVG": expected}


@pytest.mark.parametrize("value", [True, 7.0, "good", {"answer": [1, True]}])
def test_assessment_grouped_values(smoke, value):
    eid = smoke.experiment()
    info = smoke.trace_info(eid)
    for _ in range(2):
        created = smoke.rest.create_assessment(
            Feedback(name="quality", value=value, trace_id=info.trace_id)
        )
        assert_native_value(
            smoke.rest.get_assessment(info.trace_id, created.assessment_id).value, value
        )
        assert_native_value(
            smoke.document("assessments", {"assessment_id": created.assessment_id})["content"][
                "feedback"
            ]["value"],
            value,
        )
    result = query(
        smoke,
        eid,
        "ASSESSMENTS",
        "assessment_count",
        dimensions=["assessment_value"],
        filters=["assessment.name = 'quality'"],
    )
    assert len(result) == 1
    assert result[0].values == {"COUNT": 2}
    assert set(result[0].dimensions) == {"assessment_value"}
    assert_native_value(json.loads(result[0].dimensions["assessment_value"]), value)


def test_assessment_numeric_average_excludes_overrides(smoke):
    eid = smoke.experiment()
    info = smoke.trace_info(eid)
    original = smoke.rest.create_assessment(
        Feedback(name="quality", value=False, trace_id=info.trace_id)
    )
    assert smoke.rest.get_assessment(info.trace_id, original.assessment_id).value is False
    smoke.document("assessments", {"assessment_id": original.assessment_id})
    override = smoke.rest.create_assessment(
        Feedback(
            name="quality", value=True, trace_id=info.trace_id, overrides=original.assessment_id
        )
    )
    assert smoke.rest.get_assessment(info.trace_id, original.assessment_id).valid is False
    assert (
        smoke.document("assessments", {"assessment_id": original.assessment_id})["content"]["valid"]
        is False
    )
    for value in ("no", 3, "unscored"):
        created = smoke.rest.create_assessment(
            Feedback(name="quality", value=value, trace_id=info.trace_id)
        )
        assert smoke.rest.get_assessment(info.trace_id, created.assessment_id).value == value
        assert (
            smoke.document("assessments", {"assessment_id": created.assessment_id})["content"][
                "feedback"
            ]["value"]
            == value
        )
    result = query(
        smoke,
        eid,
        "ASSESSMENTS",
        "assessment_value",
        "AVG",
        filters=["assessment.name = 'quality'"],
    )
    assert result[0].values["AVG"] == pytest.approx(4 / 3)
    smoke.rest.delete_assessment(info.trace_id, override.assessment_id)
    assert smoke.rest.get_assessment(info.trace_id, original.assessment_id).valid is True
    assert smoke.documents("assessments", {"assessment_id": override.assessment_id}) == []
    result = query(
        smoke,
        eid,
        "ASSESSMENTS",
        "assessment_value",
        "AVG",
        filters=["assessment.name = 'quality'"],
    )
    assert result[0].values["AVG"] == 1
