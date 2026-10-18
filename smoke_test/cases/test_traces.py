import inspect
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import mlflow
from mlflow.entities import TraceState
from mlflow.entities.span import Span
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils.search_utils import SearchTraceUtils
import pytest

from smoke_test.support import error, kv, pages, span

pytestmark = pytest.mark.family("traces")


def optional_case(expression, expected, available, reason):
    marks = [] if available else [pytest.mark.upstream(reason), pytest.mark.skip(reason=reason)]
    return pytest.param(expression, expected, marks=marks)


def test_simple_attributes_artifacts_and_restart(smoke):
    trace = smoke.trace(
        attributes={
            "model": "fake-model",
            "temperature": 0.2,
            "provider": "test",
            "structured": {"list": [1, True]},
        }
    )
    root = trace.data.spans[0]
    assert root.get_attribute("model") == "fake-model"
    assert root.get_attribute("structured") == {"list": [1, True]}
    assert root.outputs == {"answer": 4}
    document = smoke.document("traces", {"_id": trace.info.trace_id})
    tags = kv(document["tags"])
    assert trace.info.tags == tags
    location = tags.get("mlflow.trace.spansLocation", "ARTIFACT_REPO")
    if location == "ARTIFACT_REPO":
        artifact_uri = tags["mlflow.artifactLocation"]
        path = Path(unquote(urlsplit(artifact_uri).path)) / "traces.json"
        payload = json.loads(path.read_text())
        assert len(payload["spans"]) == 1
        assert payload["spans"][0]["name"] == root.name
        smoke.record("artifact", uri=artifact_uri, path=str(path), payload=payload)
    else:
        assert location == "TRACKING_STORE"
        if hasattr(smoke.rest, "get_trace"):
            loaded = smoke.rest.get_trace(trace.info.trace_id)
        else:
            records = smoke.rest.batch_get_traces([trace.info.trace_id])
            assert len(records) == 1
            loaded = records[0]
        assert loaded.info.trace_id == trace.info.trace_id
        assert len(loaded.data.spans) == 1
    stored = smoke.documents("spans", {"trace_id": trace.info.trace_id})
    assert len(stored) == 1
    persisted_root = Span.from_dict(stored[0]["content"])
    assert persisted_root.get_attribute("model") == "fake-model"
    assert persisted_root.get_attribute("structured") == {"list": [1, True]}
    smoke.restart()
    smoke.fresh_read(
        "from mlflow import MlflowClient; "
        f"t=MlflowClient().get_trace({trace.info.trace_id!r}); "
        "assert len(t.data.spans)==1; assert t.data.spans[0].outputs=={'answer':4}; "
        "assert t.data.spans[0].get_attribute('structured')=={'list':[1,True]}"
    )


def test_decorated_nested_script_equivalent(smoke):
    eid = smoke.experiment()
    mlflow.set_experiment(experiment_id=eid)

    @mlflow.trace
    def fake_llm(question):
        return "hello " + question

    @mlflow.trace
    def process_question(question):
        return fake_llm(question)

    assert process_question("Ada") == "hello Ada"
    mlflow.flush_trace_async_logging()
    smoke.wait_trace(mlflow.get_last_active_trace_id(), expected_spans=2)
    found = smoke.search_traces(eid)
    assert len(found) == 1
    trace = smoke.client.get_trace(found[0].info.trace_id)
    spans = {item.name: item for item in trace.data.spans}
    assert set(spans) == {"process_question", "fake_llm"}
    assert spans["fake_llm"].parent_id == spans["process_question"].span_id
    assert spans["fake_llm"].outputs == "hello Ada"
    assert smoke.document("traces", {"_id": trace.info.trace_id})["state"] == "OK"
    docs = smoke.documents("spans", {"trace_id": trace.info.trace_id})
    assert {item["span_id"] for item in docs} == {item.span_id for item in spans.values()}
    assert len(docs) == 2


def test_error_script_equivalent(smoke):
    eid = smoke.experiment()
    mlflow.set_experiment(experiment_id=eid)
    with pytest.raises(ValueError, match="intentional smoke exception"):
        with mlflow.start_span(name="explicit_error_test") as current:
            trace_id = current.trace_id
            raise ValueError("intentional smoke exception")
    mlflow.flush_trace_async_logging()
    trace = smoke.wait_trace(trace_id)
    assert trace.info.state == TraceState.ERROR
    assert any(event.name == "exception" for event in trace.data.spans[0].events)
    assert smoke.document("traces", {"_id": trace_id})["state"] == "ERROR"
    stored = smoke.documents("spans", {"trace_id": trace_id})
    assert len(stored) == 1
    persisted_span = Span.from_dict(stored[0]["content"])
    assert any(
        event.name == "exception"
        and event.attributes.get("exception.message") == "intentional smoke exception"
        for event in persisted_span.events
    )


def test_experiment_isolation_script_equivalent(smoke):
    first = smoke.experiment("first")
    second = smoke.experiment("second")
    a = smoke.trace(first, "first-trace")
    b = smoke.trace(second, "second-trace")
    assert [item.info.trace_id for item in smoke.search_traces(first)] == [a.info.trace_id]
    assert [item.info.trace_id for item in smoke.search_traces(second)] == [b.info.trace_id]
    assert len(smoke.documents("traces", {"experiment_id": first})) == 1
    assert len(smoke.documents("traces", {"experiment_id": second})) == 1


def test_trace_metadata_transport(smoke):
    eid = smoke.experiment()
    info = smoke.trace_info(
        eid,
        tags={"team.name": "$risk"},
        trace_metadata={"session.key": "$session"},
        client_request_id="client-smoke",
        request_preview="question",
        response_preview="answer",
    )
    loaded = smoke.rest.get_trace_info(info.trace_id)
    assert loaded.client_request_id == "client-smoke"
    assert loaded.request_preview == "question" and loaded.response_preview == "answer"
    doc = smoke.document("traces", {"_id": info.trace_id})
    assert kv(doc["tags"])["team.name"] == loaded.tags["team.name"] == "$risk"
    assert (
        kv(doc["trace_metadata"])["session.key"]
        == loaded.trace_metadata["session.key"]
        == "$session"
    )
    assert doc["client_request_id"] == "client-smoke"


def test_trace_tag_lifecycle(smoke):
    trace = smoke.trace()
    tid = trace.info.trace_id
    for value in ("old", "$new"):
        smoke.client.set_trace_tag(tid, "team.name", value)
        assert (
            smoke.rest.get_trace_info(tid).tags["team.name"]
            == kv(smoke.document("traces", {"_id": tid})["tags"])["team.name"]
            == value
        )
    smoke.client.delete_trace_tag(tid, "team.name")
    assert "team.name" not in smoke.rest.get_trace_info(tid).tags
    assert "team.name" not in kv(smoke.document("traces", {"_id": tid})["tags"])
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.delete_trace_tag(tid, "missing")
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.set_trace_tag("tr-" + "f" * 32, "tag", "value")
    assert smoke.documents("traces", {"_id": "tr-" + "f" * 32}) == []


def test_child_first_span_ingestion_and_redelivery(smoke):
    eid = smoke.experiment()
    tid = "tr-" + uuid4().hex
    child = span(tid, number=2, parent=1)
    smoke.rest.log_spans(eid, [child])
    assert smoke.rest.get_trace_info(tid).state == TraceState.IN_PROGRESS
    assert len(smoke.documents("spans", {"trace_id": tid})) == 1
    root = span(tid, attributes={"smoke.attribute": json.dumps({"literal": "$value"})})
    smoke.rest.log_spans(eid, [root])
    assert smoke.rest.get_trace_info(tid).state == TraceState.OK
    before = smoke.documents("spans", {"trace_id": tid})
    assert len(before) == 2
    smoke.rest.log_spans(eid, [child, root])
    assert smoke.rest.get_trace_info(tid).state == TraceState.OK
    assert smoke.documents("spans", {"trace_id": tid}) == before
    if "batch_get_traces" in type(smoke.rest).__dict__:
        records = smoke.rest.batch_get_traces([tid])
        assert len(records) == 1 and len(records[0].data.spans) == 2


@pytest.mark.upstream("batch_get_traces REST transport is absent in MLflow 3.5")
@pytest.mark.skipif(
    "batch_get_traces" not in RestStore.__dict__, reason="No full-trace REST transport"
)
def test_batch_reads(smoke):
    eid = smoke.experiment()
    a, b = smoke.trace(eid, "a"), smoke.trace(eid, "b")
    other = smoke.trace(smoke.experiment("other"))
    ids = [b.info.trace_id, "tr-" + "f" * 32, a.info.trace_id]
    found = smoke.rest.batch_get_traces(ids)
    assert [item.info.trace_id for item in found] == [b.info.trace_id, a.info.trace_id]
    if "experiment_ids" in inspect.signature(smoke.rest.batch_get_traces).parameters:
        assert smoke.rest.batch_get_traces([other.info.trace_id], experiment_ids=[eid]) == []
    assert len(smoke.documents("traces", {"experiment_id": eid})) == 2


@pytest.mark.parametrize("by_timestamp", [False, True], ids=["ids", "timestamp-count"])
def test_trace_deletion_cascades(smoke, by_timestamp):
    from mlflow.entities.assessment import Feedback

    eid = smoke.experiment()
    first = smoke.trace(eid, "first")
    second = smoke.trace(eid, "second")
    assessment = smoke.rest.create_assessment(
        Feedback(name="quality", value=True, trace_id=first.info.trace_id)
    )
    assert smoke.rest.get_assessment(first.info.trace_id, assessment.assessment_id).value is True
    smoke.document("assessments", {"assessment_id": assessment.assessment_id})
    kwargs = (
        {"max_timestamp_millis": second.info.request_time + 10000, "max_traces": 1}
        if by_timestamp
        else {"trace_ids": [first.info.trace_id]}
    )
    smoke.client.delete_traces(experiment_id=eid, **kwargs)
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.rest.get_trace_info(first.info.trace_id)
    for collection, query in [
        ("traces", {"_id": first.info.trace_id}),
        ("spans", {"trace_id": first.info.trace_id}),
        ("assessments", {"trace_id": first.info.trace_id}),
    ]:
        assert smoke.documents(collection, query) == []
    assert smoke.client.get_trace(second.info.trace_id).info.trace_id == second.info.trace_id
    assert len(smoke.documents("spans", {"trace_id": second.info.trace_id})) == 1


@pytest.fixture
def searchable_traces(smoke):
    eid = smoke.experiment()
    result = {}
    for name, timestamp, state, tags, metadata in [
        (
            "a",
            1000,
            TraceState.OK,
            {"mlflow.traceName": "alpha", "team.name": "$risk"},
            {"session.key": "one"},
        ),
        (
            "b",
            2000,
            TraceState.ERROR,
            {"mlflow.traceName": "beta", "team.name": "other"},
            {TraceMetadataKey.SOURCE_RUN: "run"},
        ),
        ("c", 3000, TraceState.IN_PROGRESS, {}, {}),
    ]:
        result[name] = smoke.trace_info(
            eid,
            request_time=timestamp,
            state=state,
            tags=tags,
            trace_metadata=metadata,
            client_request_id=name,
        ).trace_id
    return eid, result


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("status = 'ERROR'", ["b"]),
        ("status != 'ERROR'", ["a", "c"]),
        ("timestamp_ms >= 2000", ["b", "c"]),
        ("tags.`team.name` = '$risk'", ["a"]),
        ("request_metadata.`session.key` = 'one'", ["a"]),
        ("run_id = 'run'", ["b"]),
        ("status = 'OK' AND tags.`team.name` = '$risk'", ["a"]),
        optional_case(
            "name ILIKE 'AL%'",
            ["a"],
            "ILIKE" in SearchTraceUtils.VALID_TAG_COMPARATORS,
            "Trace ILIKE absent upstream",
        ),
        optional_case(
            "name RLIKE '^b.*a$'",
            ["b"],
            "RLIKE" in SearchTraceUtils.VALID_TAG_COMPARATORS,
            "Trace RLIKE absent upstream",
        ),
        optional_case(
            "tags.`team.name` IS NULL",
            ["c"],
            "IS NULL" in SearchTraceUtils.VALID_TAG_COMPARATORS,
            "Trace existence filters absent upstream",
        ),
        optional_case(
            "client_request_id = 'a'",
            ["a"],
            "client_request_id" in SearchTraceUtils.VALID_SEARCH_ATTRIBUTE_KEYS,
            "client_request_id filter absent upstream",
        ),
        optional_case(
            "end_time_ms >= 2010",
            ["b", "c"],
            "end_time_ms" in SearchTraceUtils.VALID_SEARCH_ATTRIBUTE_KEYS,
            "End-time trace filters absent upstream",
        ),
    ],
)
def test_trace_search_filters(smoke, searchable_traces, expression, expected):
    eid, traces = searchable_traces
    before = smoke.documents("traces", {})
    found = pages(
        lambda token: smoke.search_traces(
            eid,
            filter_string=expression,
            include_spans=False,
            order_by=["timestamp_ms ASC"],
            max_results=1,
            page_token=token,
        ),
        lambda item: item.info.trace_id,
    )
    assert [item.info.trace_id for item in found] == [traces[name] for name in expected]
    assert smoke.documents("traces", {}) == before


@pytest.mark.parametrize(
    "direction,expected", [("ASC", ["a", "b", "c"]), ("DESC", ["c", "b", "a"])]
)
def test_trace_search_order(smoke, searchable_traces, direction, expected):
    eid, traces = searchable_traces
    found = pages(
        lambda token: smoke.search_traces(
            eid,
            include_spans=False,
            order_by=[f"timestamp_ms {direction}"],
            max_results=1,
            page_token=token,
        ),
        lambda item: item.info.trace_id,
    )
    assert [item.info.trace_id for item in found] == [traces[name] for name in expected]


@pytest.mark.parametrize(
    "kwargs", [{"filter_string": "unknown = 'x'"}, {"page_token": "bad-token"}, {"max_results": 0}]
)
def test_invalid_trace_search(smoke, kwargs):
    eid = smoke.experiment()
    with error("INVALID_PARAMETER_VALUE"):
        smoke.rest.search_traces(experiment_ids=[eid], **kwargs)


def test_span_usage_deduplication_and_authoritative_metadata(smoke):
    eid = smoke.experiment()
    tid = "tr-" + uuid4().hex
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    child = span(tid, 2, parent=1, attributes={SpanAttributeKey.CHAT_USAGE: json.dumps(usage)})
    root = span(tid, attributes={SpanAttributeKey.CHAT_USAGE: json.dumps(usage)})
    smoke.rest.log_spans(eid, [child, root])
    loaded = smoke.rest.get_trace_info(tid)
    assert json.loads(loaded.trace_metadata[TraceMetadataKey.TOKEN_USAGE])["input_tokens"] == 10
    assert smoke.document("traces", {"_id": tid})["token_usage"]["input_tokens"] == 10
    assert len(smoke.documents("spans", {"trace_id": tid})) == 2
    authoritative = smoke.trace_info(
        eid, trace_metadata={TraceMetadataKey.TOKEN_USAGE: '{"input_tokens": 42}'}
    )
    smoke.rest.log_spans(
        eid,
        [span(authoritative.trace_id, attributes={SpanAttributeKey.CHAT_USAGE: json.dumps(usage)})],
    )
    assert json.loads(
        smoke.rest.get_trace_info(authoritative.trace_id).trace_metadata[
            TraceMetadataKey.TOKEN_USAGE
        ]
    ) == {"input_tokens": 42}
    assert smoke.document("traces", {"_id": authoritative.trace_id})["token_usage"] == {
        "input_tokens": 42
    }


@pytest.mark.upstream("Batch trace info route not exposed by this MLflow version")
@pytest.mark.skipif(
    "batch_get_trace_infos" not in RestStore.__dict__,
    reason="No batch trace-info HTTP transport upstream",
)
def test_batch_info_reads_order_and_scope(smoke):
    eid = smoke.experiment()
    first, second = smoke.trace_info(eid), smoke.trace_info(eid)
    other = smoke.trace_info(smoke.experiment("other"))
    kwargs = (
        {"experiment_ids": [eid]}
        if "experiment_ids" in inspect.signature(smoke.rest.batch_get_trace_infos).parameters
        else {}
    )
    requested = [second.trace_id, "tr-" + "f" * 32, first.trace_id, second.trace_id]
    found = smoke.rest.batch_get_trace_infos(requested, **kwargs)
    assert [item.trace_id for item in found] == [second.trace_id, first.trace_id]
    if kwargs:
        assert smoke.rest.batch_get_trace_infos([other.trace_id], **kwargs) == []
    assert len(smoke.documents("traces", {"experiment_id": eid})) == 2


@pytest.mark.upstream("Full-trace batch HTTP route unavailable upstream")
@pytest.mark.skipif(
    "batch_get_traces" not in RestStore.__dict__, reason="No batch trace HTTP transport upstream"
)
def test_partial_and_empty_trace_reads(smoke):
    eid = smoke.experiment()
    empty = smoke.trace_info(eid)
    partial = smoke.trace_info(
        eid, trace_metadata={TraceMetadataKey.SIZE_STATS: '{"num_spans": 2}'}
    )
    smoke.rest.log_spans(eid, [span(partial.trace_id)])
    assert smoke.rest.get_trace_info(partial.trace_id).trace_id == partial.trace_id
    assert len(smoke.documents("spans", {"trace_id": partial.trace_id})) == 1
    complete = smoke.trace(eid)
    result = smoke.rest.batch_get_traces([empty.trace_id, partial.trace_id, complete.info.trace_id])
    assert [item.info.trace_id for item in result] == [complete.info.trace_id]
    get_trace = getattr(smoke.rest, "get_trace", None)
    if get_trace is not None and "allow_partial" in inspect.signature(get_trace).parameters:
        assert smoke.rest.get_trace(empty.trace_id, allow_partial=True).data.spans == []
        assert len(smoke.rest.get_trace(partial.trace_id, allow_partial=True).data.spans) == 1


def test_span_tags_preserve_storage_location(smoke):
    eid = smoke.experiment()
    info = smoke.trace_info(eid)
    original = dict(info.tags)
    smoke.rest.log_spans(
        eid,
        [
            span(
                info.trace_id,
                attributes={
                    "mlflow.traceTag.mlflow.trace.spansLocation": "UNTRUSTED",
                    "mlflow.traceTag.mlflow.artifactLocation": "s3://untrusted",
                    "mlflow.traceTag.team.name": "$risk",
                },
            )
        ],
    )
    loaded = smoke.rest.get_trace_info(info.trace_id)
    document = smoke.document("traces", {"_id": info.trace_id})
    tags = kv(document["tags"])
    assert (
        loaded.tags["mlflow.artifactLocation"]
        == tags["mlflow.artifactLocation"]
        == original["mlflow.artifactLocation"]
    )
    assert (
        loaded.tags["mlflow.trace.spansLocation"]
        == tags["mlflow.trace.spansLocation"]
        == original["mlflow.trace.spansLocation"]
    )
    assert loaded.tags["team.name"] == tags["team.name"] == "$risk"


def test_custom_trace_name_order_missing_last(smoke, searchable_traces):
    eid, traces = searchable_traces
    result = pages(
        lambda token: smoke.search_traces(
            eid, include_spans=False, order_by=["name DESC"], max_results=1, page_token=token
        ),
        lambda item: item.info.trace_id,
    )
    assert [item.info.trace_id for item in result] == [traces[name] for name in ("b", "a", "c")]


def test_delete_trace_ids_cannot_cross_experiment_scope(smoke):
    first_eid, second_eid = smoke.experiment("first"), smoke.experiment("second")
    first, second = smoke.trace(first_eid), smoke.trace(second_eid)
    smoke.client.delete_traces(
        experiment_id=first_eid, trace_ids=[first.info.trace_id, second.info.trace_id]
    )
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.rest.get_trace_info(first.info.trace_id)
    assert smoke.documents("traces", {"_id": first.info.trace_id}) == []
    assert smoke.documents("spans", {"trace_id": first.info.trace_id}) == []
    assert smoke.client.get_trace(second.info.trace_id).info.trace_id == second.info.trace_id
    assert smoke.document("traces", {"_id": second.info.trace_id})["experiment_id"] == second_eid
    assert len(smoke.documents("spans", {"trace_id": second.info.trace_id})) == 1
