import math

import mlflow
from mlflow.entities import (
    Dataset,
    DatasetInput,
    InputTag,
    LoggedModelInput,
    LoggedModelOutput,
    Metric,
    Param,
    RunTag,
    ViewType,
)
from mlflow.utils.search_utils import SearchUtils
from mlflow.exceptions import MlflowException
import pytest

from smoke_test.support import error, kv, pages

pytestmark = pytest.mark.family("runs")


def test_rest_create_update_and_individual_logging(smoke):
    eid = smoke.experiment()
    response = smoke.http(
        "POST",
        "runs/create",
        {
            "experiment_id": eid,
            "start_time": 1000,
            "run_name": "rest-run",
            "tags": [{"key": "origin", "value": "REST"}],
        },
    )
    rid = response["run"]["info"]["run_id"]
    assert smoke.client.get_run(rid).info.run_name == "rest-run"
    assert smoke.document("runs", {"_id": rid})["start_time"] == 1000
    smoke.client.log_param(rid, "depth", "3", synchronous=True)
    assert (
        smoke.client.get_run(rid).data.params["depth"]
        == kv(smoke.document("runs", {"_id": rid})["params"])["depth"]
        == "3"
    )
    smoke.client.log_param(rid, "depth", "3", synchronous=True)
    assert (
        smoke.client.get_run(rid).data.params
        == kv(smoke.document("runs", {"_id": rid})["params"])
        == {"depth": "3"}
    )
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.log_param(rid, "depth", "4", synchronous=True)
    assert (
        smoke.client.get_run(rid).data.params
        == kv(smoke.document("runs", {"_id": rid})["params"])
        == {"depth": "3"}
    )
    smoke.client.log_metric(rid, "loss", 1.5, timestamp=1010, step=1, synchronous=True)
    assert (
        smoke.client.get_run(rid).data.metrics
        == kv(smoke.document("runs", {"_id": rid})["metrics"])
        == {"loss": 1.5}
    )
    smoke.client.set_tag(rid, "team.name", "$risk", synchronous=True)
    assert (
        smoke.client.get_run(rid).data.tags["team.name"]
        == kv(smoke.document("runs", {"_id": rid})["tags"])["team.name"]
        == "$risk"
    )
    smoke.http(
        "POST",
        "runs/update",
        {"run_id": rid, "status": "FINISHED", "end_time": 2000, "run_name": "$renamed"},
    )
    loaded = smoke.client.get_run(rid)
    doc = smoke.document("runs", {"_id": rid})
    assert (loaded.info.status, loaded.info.end_time, loaded.info.run_name) == (
        "FINISHED",
        2000,
        "$renamed",
    )
    assert (doc["status"], doc["end_time"], doc["name"]) == ("FINISHED", 2000, "$renamed")
    assert kv(doc["tags"])["mlflow.runName"] == loaded.data.tags["mlflow.runName"] == "$renamed"


@pytest.mark.parametrize("synchronous", [True, False], ids=["sync", "async"])
def test_batch_logging(smoke, synchronous):
    rid = smoke.run().info.run_id
    pending = smoke.client.log_batch(
        rid,
        metrics=[Metric("loss", 0.5, 1000, 1)],
        params=[Param("depth", "3")],
        tags=[RunTag("keep", "yes"), RunTag("replace", "old")],
        synchronous=synchronous,
    )
    if pending is not None:
        pending.wait()
    loaded = smoke.client.get_run(rid)
    doc = smoke.document("runs", {"_id": rid})
    assert loaded.data.params == kv(doc["params"]) == {"depth": "3"}
    assert loaded.data.metrics == kv(doc["metrics"]) == {"loss": 0.5}
    assert loaded.data.tags["replace"] == kv(doc["tags"])["replace"] == "old"
    pending = smoke.client.log_batch(rid, tags=[RunTag("replace", "$new")], synchronous=synchronous)
    if pending is not None:
        pending.wait()
    assert (
        smoke.client.get_run(rid).data.tags["replace"]
        == kv(smoke.document("runs", {"_id": rid})["tags"])["replace"]
        == "$new"
    )
    assert len(smoke.documents("run_metrics", {"run_id": rid, "k": "loss"})) == 1


def test_async_param_error_propagates(smoke):
    rid = smoke.run().info.run_id
    smoke.client.log_param(rid, "fixed", "a", synchronous=True)
    assert (
        smoke.client.get_run(rid).data.params
        == kv(smoke.document("runs", {"_id": rid})["params"])
        == {"fixed": "a"}
    )
    pending = smoke.client.log_param(rid, "fixed", "b", synchronous=False)
    with pytest.raises(MlflowException) as caught:
        pending.wait()
    assert "param" in str(caught.value).lower()
    assert (
        smoke.client.get_run(rid).data.params
        == kv(smoke.document("runs", {"_id": rid})["params"])
        == {"fixed": "a"}
    )


def test_history_latest_and_pages(smoke):
    rid = smoke.run().info.run_id
    events = [
        Metric("loss", 0.4, 1000, 1),
        Metric("loss", 0.9, 900, 2),
        Metric("loss", 0.8, 1001, 1),
        Metric("loss", 0.6, 950, 2),
        Metric("loss", 0.7, 950, 2),
    ]
    smoke.client.log_batch(rid, metrics=events, synchronous=True)
    assert (
        smoke.client.get_run(rid).data.metrics
        == kv(smoke.document("runs", {"_id": rid})["metrics"])
        == {"loss": 0.7}
    )
    history = pages(
        lambda token: smoke.rest.get_metric_history(rid, "loss", max_results=2, page_token=token),
        lambda metric: (metric.timestamp, metric.step, metric.value),
    )
    expected = sorted((m.timestamp, m.step, m.value) for m in events)
    assert [(m.timestamp, m.step, m.value) for m in history] == expected
    stored = smoke.documents("run_metrics", {"run_id": rid, "k": "loss"})
    assert sorted((m["timestamp"], m["step"], m["v"]) for m in stored) == expected
    assert smoke.client.get_metric_history(rid, "absent") == []


def test_nonfinite_metric_values(smoke):
    rid = smoke.run().info.run_id
    smoke.client.log_batch(
        rid,
        metrics=[
            Metric("nan", float("nan"), 1000, 0),
            Metric("positive", float("inf"), 1000, 0),
            Metric("negative", -float("inf"), 1000, 0),
        ],
        synchronous=True,
    )
    values = smoke.client.get_run(rid).data.metrics
    stored = kv(smoke.document("runs", {"_id": rid})["metrics"])
    assert math.isnan(values["nan"]) and math.isnan(stored["nan"])
    assert values["positive"] == stored["positive"] == float("inf")
    assert values["negative"] == stored["negative"] == -float("inf")
    assert math.isnan(smoke.client.get_metric_history(rid, "nan")[0].value)


def test_dataset_inputs_and_fluent_script_equivalent(smoke):
    import numpy as np

    eid = smoke.experiment()
    mlflow.set_experiment(experiment_id=eid)
    dataset = mlflow.data.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]), targets=np.array([0, 1]), name="smoke-dataset"
    )
    with mlflow.start_run(run_name="mongo-smoke-dataset-inputs") as run:
        mlflow.log_input(dataset, context="training", tags={"literal.key": "$value"})
        rid = run.info.run_id
        loaded = smoke.client.get_run(rid).inputs.dataset_inputs
        doc = smoke.document("runs", {"_id": rid})
        assert (
            loaded[0].dataset.digest
            == doc["inputs"]["datasets"][0]["dataset"]["digest"]
            == dataset.digest
        )
        assert kv(doc["inputs"]["datasets"][0]["tags"]) == {
            "literal.key": "$value",
            "mlflow.data.context": "training",
        }
        mlflow.log_input(dataset, context="training", tags={"literal.key": "$value"})
        assert (
            len(smoke.client.get_run(rid).inputs.dataset_inputs)
            == len(smoke.document("runs", {"_id": rid})["inputs"]["datasets"])
            == 1
        )
    assert (
        smoke.client.get_run(rid).info.status
        == smoke.document("runs", {"_id": rid})["status"]
        == "FINISHED"
    )
    assert len(smoke.client.search_runs([eid], "datasets.context = 'training'")) == 1


def test_model_inputs_outputs(smoke):
    run = smoke.run()
    rid, eid = run.info.run_id, run.info.experiment_id
    model = smoke.client.create_logged_model(eid, name="input-model", source_run_id=rid)
    assert (
        smoke.client.get_logged_model(model.model_id).source_run_id
        == smoke.document("logged_models", {"_id": model.model_id})["source_run_id"]
        == rid
    )
    smoke.client.log_inputs(rid, models=[LoggedModelInput(model_id=model.model_id)])
    assert (
        smoke.client.get_run(rid).inputs.model_inputs[0].model_id
        == smoke.document("runs", {"_id": rid})["inputs"]["models"][0]["model_id"]
        == model.model_id
    )
    smoke.client.log_outputs(rid, [LoggedModelOutput(model.model_id, 2)])
    assert (
        smoke.client.get_run(rid).outputs.model_outputs[0].step
        == smoke.document("runs", {"_id": rid})["outputs"][0]["step"]
        == 2
    )
    smoke.client.log_outputs(rid, [LoggedModelOutput(model.model_id, 2)])
    assert (
        len(smoke.client.get_run(rid).outputs.model_outputs)
        == len(smoke.document("runs", {"_id": rid})["outputs"])
        == 2
    )


def test_run_delete_restore(smoke):
    run = smoke.run()
    rid, eid = run.info.run_id, run.info.experiment_id
    smoke.client.log_metric(rid, "loss", 0.5, synchronous=True)
    assert (
        smoke.client.get_run(rid).data.metrics
        == kv(smoke.document("runs", {"_id": rid})["metrics"])
        == {"loss": 0.5}
    )
    smoke.client.delete_run(rid)
    assert (
        smoke.client.get_run(rid).info.lifecycle_stage
        == smoke.document("runs", {"_id": rid})["lifecycle_stage"]
        == "deleted"
    )
    assert smoke.document("runs", {"_id": rid})["deleted_time"] > 0
    assert smoke.client.search_runs([eid]) == []
    assert [
        item.info.run_id
        for item in smoke.client.search_runs([eid], run_view_type=ViewType.DELETED_ONLY)
    ] == [rid]
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.log_metric(rid, "forbidden", 1.0, synchronous=True)
    smoke.client.restore_run(rid)
    assert (
        smoke.client.get_run(rid).info.lifecycle_stage
        == smoke.document("runs", {"_id": rid})["lifecycle_stage"]
        == "active"
    )
    assert smoke.document("runs", {"_id": rid})["deleted_time"] is None
    assert smoke.client.get_metric_history(rid, "loss")[0].value == 0.5


@pytest.fixture
def searchable_runs(smoke):
    eid = smoke.experiment()
    result = {}
    for name, value, depth, priority in [("low", 1.0, "10", "b"), ("high", 2.0, "2", "a")]:
        run = smoke.run(eid, name, start_time=1000)
        rid = run.info.run_id
        smoke.client.log_batch(
            rid,
            metrics=[Metric("loss", value, 1000, 1)],
            params=[Param("depth", depth), Param("empty", "")],
            tags=[RunTag("priority", priority), RunTag("team.name", "Risk")],
            synchronous=True,
        )
        loaded = smoke.client.get_run(rid)
        doc = smoke.document("runs", {"_id": rid})
        assert loaded.data.metrics == kv(doc["metrics"]) == {"loss": value}
        assert loaded.data.params == kv(doc["params"]) == {"depth": depth, "empty": ""}
        assert loaded.data.tags["priority"] == kv(doc["tags"])["priority"] == priority
        result[name] = rid
    result["missing"] = smoke.run(eid, "missing", start_time=1000).info.run_id
    return eid, result


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("metrics.loss > 1", {"high"}),
        ("metrics.loss >= 1", {"low", "high"}),
        ("metrics.loss < 2", {"low"}),
        ("metrics.loss <= 2", {"low", "high"}),
        ("metrics.loss = 1", {"low"}),
        ("metrics.loss != 1", {"high"}),
        ("params.depth = '10'", {"low"}),
        ("params.depth != '10'", {"high"}),
        ("params.depth LIKE '1%'", {"low"}),
        ("tags.priority != 'a'", {"low"}),
        ("tags.`team.name` ILIKE 'risk'", {"low", "high"}),
        ("tags.`team.name` LIKE 'risk'", set()),
        ("attributes.run_name = 'high'", {"high"}),
        ("attributes.start_time >= 1000", {"low", "high", "missing"}),
        ("metrics.loss >= 1 AND params.depth = '2' AND tags.priority = 'a'", {"high"}),
        ("tags.priority = 'a' AND tags.priority = 'b'", set()),
        ("params.empty = ''", {"low", "high"}),
    ],
)
def test_run_search_filters(smoke, searchable_runs, expression, expected):
    eid, runs = searchable_runs
    before = smoke.documents("runs", {})
    found = pages(
        lambda token: smoke.client.search_runs(
            [eid], expression, order_by=["attributes.run_name ASC"], max_results=1, page_token=token
        ),
        lambda item: item.info.run_id,
    )
    assert [item.info.run_name for item in found] == sorted(expected)
    assert {item.info.run_id for item in found} == {runs[name] for name in expected}
    assert smoke.documents("runs", {}) == before


@pytest.mark.parametrize(
    "direction,expected",
    [("ASC", ["low", "high", "missing"]), ("DESC", ["high", "low", "missing"])],
)
def test_run_metric_sort(smoke, searchable_runs, direction, expected):
    eid, _ = searchable_runs
    found = pages(
        lambda token: smoke.client.search_runs(
            [eid], order_by=[f"metrics.loss {direction}"], max_results=1, page_token=token
        ),
        lambda item: item.info.run_id,
    )
    assert [item.info.run_name for item in found] == expected


def test_run_membership_and_dataset_search(smoke):
    run = smoke.run()
    eid, rid = run.info.experiment_id, run.info.run_id
    datasets = [
        DatasetInput(
            Dataset("training", "aaa", "smoke", "{}"), [InputTag("mlflow.data.context", "train")]
        ),
        DatasetInput(
            Dataset("evaluation", "bbb", "smoke", "{}"), [InputTag("mlflow.data.context", "eval")]
        ),
    ]
    smoke.client.log_inputs(rid, datasets=datasets)
    assert (
        len(smoke.client.get_run(rid).inputs.dataset_inputs)
        == len(smoke.document("runs", {"_id": rid})["inputs"]["datasets"])
        == 2
    )
    for expression in [
        f"run_id IN ('{rid}')",
        "datasets.name = 'training'",
        "datasets.digest = 'bbb'",
        "datasets.context = 'eval'",
        "datasets.name = 'training' AND datasets.digest = 'bbb'",
    ]:
        assert [item.info.run_id for item in smoke.client.search_runs([eid], expression)] == [rid]
    assert smoke.client.search_runs([eid], f"run_id NOT IN ('{rid}')") == []
    assert smoke.client.search_runs([eid], "datasets.context = 'missing'") == []
    assert smoke.client.search_runs([smoke.experiment("other")]) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"filter_string": "metrics.loss LIKE 'x'"},
        {"page_token": "bad-token"},
        {"max_results": 0},
        {"order_by": ["bad ASC"]},
    ],
)
def test_invalid_run_search(smoke, kwargs):
    eid = smoke.experiment()
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.search_runs([eid], **kwargs)


@pytest.mark.parametrize(
    "category,comparator,expected",
    [
        pytest.param(
            category,
            comparator,
            expected,
            marks=[]
            if comparator in supported
            else [
                pytest.mark.upstream("Run key existence comparator absent upstream"),
                pytest.mark.skip(reason="Run key existence comparator absent upstream"),
            ],
        )
        for category, supported in [
            ("params", SearchUtils.VALID_PARAM_COMPARATORS),
            ("tags", SearchUtils.VALID_TAG_COMPARATORS),
        ]
        for comparator, expected in [("IS NULL", {"missing"}), ("IS NOT NULL", {"present"})]
    ],
)
def test_run_key_existence(smoke, category, comparator, expected):
    eid = smoke.experiment()
    first = smoke.run(eid, "present")
    smoke.run(eid, "missing")
    smoke.client.log_batch(
        first.info.run_id, params=[Param("empty", "")], tags=[RunTag("empty", "")], synchronous=True
    )
    loaded = smoke.client.get_run(first.info.run_id)
    doc = smoke.document("runs", {"_id": first.info.run_id})
    assert loaded.data.params == kv(doc["params"]) == {"empty": ""}
    assert loaded.data.tags["empty"] == kv(doc["tags"])["empty"] == ""
    assert {
        item.info.run_name
        for item in smoke.client.search_runs([eid], f"{category}.empty {comparator}")
    } == expected


def test_search_uses_latest_summary_not_history(smoke):
    run = smoke.run()
    rid, eid = run.info.run_id, run.info.experiment_id
    smoke.client.log_batch(
        rid, metrics=[Metric("loss", 1.0, 1000, 1), Metric("loss", 9.0, 2000, 0)], synchronous=True
    )
    assert (
        smoke.client.get_run(rid).data.metrics
        == kv(smoke.document("runs", {"_id": rid})["metrics"])
        == {"loss": 1.0}
    )
    assert len(smoke.documents("run_metrics", {"run_id": rid})) == 2
    assert smoke.client.search_runs([eid], "metrics.loss > 5") == []
    assert [item.info.run_id for item in smoke.client.search_runs([eid], "metrics.loss = 1")] == [
        rid
    ]


def test_empty_batch_and_prevalidation_preserve_run(smoke):
    rid = smoke.run().info.run_id
    smoke.client.log_batch(rid, synchronous=True)
    assert (
        smoke.client.get_run(rid).data.params
        == kv(smoke.document("runs", {"_id": rid})["params"])
        == {}
    )
    before = smoke.document("runs", {"_id": rid})
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.log_batch(
            rid, params=[Param("duplicate", "a"), Param("duplicate", "b")], synchronous=True
        )
    assert smoke.client.get_run(rid).data.params == {}
    assert smoke.document("runs", {"_id": rid}) == before


def test_equal_start_time_ties_are_stable(smoke):
    eid = smoke.experiment()
    ids = [smoke.run(eid, f"run-{index}", start_time=1000).info.run_id for index in range(3)]
    result = pages(
        lambda token: smoke.client.search_runs([eid], max_results=1, page_token=token),
        lambda item: item.info.run_id,
    )
    assert [item.info.run_id for item in result] == sorted(ids)
    assert len(smoke.documents("runs", {"experiment_id": eid})) == 3
