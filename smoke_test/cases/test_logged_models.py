from mlflow.entities import Metric
import pytest

from smoke_test.support import error, kv, pages

pytestmark = pytest.mark.family("logged_models")


def model(smoke, eid, **kwargs):
    created = smoke.client.create_logged_model(eid, **kwargs)
    loaded = smoke.client.get_logged_model(created.model_id)
    doc = smoke.document("logged_models", {"_id": created.model_id})
    assert loaded.name == doc["name"] == created.name
    assert loaded.experiment_id == doc["experiment_id"] == eid
    assert loaded.params == kv(doc["params"])
    assert loaded.tags == kv(doc["tags"])
    assert loaded.artifact_location == doc["artifact_location"]
    return created


def test_logged_model_create_tags_delete(smoke):
    run = smoke.run()
    eid, rid = run.info.experiment_id, run.info.run_id
    created = model(
        smoke,
        eid,
        name="classifier",
        source_run_id=rid,
        model_type="classifier",
        tags={"keep": "yes", "team.name": "old"},
        params={"depth": "3"},
    )
    mid = created.model_id
    assert created.status.value == "PENDING"
    smoke.client.set_logged_model_tags(mid, {"team.name": "$new", "remove": "me"})
    assert (
        smoke.client.get_logged_model(mid).tags
        == kv(smoke.document("logged_models", {"_id": mid})["tags"])
        == {"keep": "yes", "team.name": "$new", "remove": "me"}
    )
    smoke.client.delete_logged_model_tag(mid, "remove")
    assert (
        smoke.client.get_logged_model(mid).tags
        == kv(smoke.document("logged_models", {"_id": mid})["tags"])
        == {"keep": "yes", "team.name": "$new"}
    )
    smoke.client.set_logged_model_tags(mid, {})
    assert smoke.client.get_logged_model(mid).tags == kv(
        smoke.document("logged_models", {"_id": mid})["tags"]
    )
    smoke.client.delete_logged_model(mid)
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_logged_model(mid)
    assert smoke.document("logged_models", {"_id": mid})["lifecycle_stage"] == "deleted"
    assert smoke.client.search_logged_models([eid]) == []
    model(smoke, eid)


@pytest.mark.parametrize("synchronous", [True, False], ids=["sync", "async"])
def test_logged_model_metric_identity(smoke, synchronous):
    run = smoke.run()
    eid, rid = run.info.experiment_id, run.info.run_id
    created = model(smoke, eid, name="scored", source_run_id=rid)
    metric = Metric(
        "loss", 0.5, 1000, 1, model_id=created.model_id, dataset_name="eval", dataset_digest="abc"
    )
    pending = smoke.client.log_batch(rid, metrics=[metric], synchronous=synchronous)
    if pending is not None:
        pending.wait()
    loaded = smoke.client.get_logged_model(created.model_id)
    assert len(loaded.metrics) == 1
    actual = loaded.metrics[0]
    assert (actual.value, actual.run_id, actual.dataset_name, actual.dataset_digest) == (
        0.5,
        rid,
        "eval",
        "abc",
    )
    doc = smoke.document("run_metrics", {"model_id": created.model_id, "run_id": rid, "k": "loss"})
    assert (doc["v"], doc["dataset_name"], doc["dataset_digest"]) == (0.5, "eval", "abc")


def test_logged_model_rest_creation_and_invalid_entries(smoke):
    eid = smoke.experiment()
    response = smoke.http(
        "POST",
        "logged-models",
        {
            "experiment_id": eid,
            "name": "rest-model",
            "params": [{"key": "depth", "value": "3"}],
            "tags": [{"key": "team", "value": "risk"}],
        },
    )
    mid = response["model"]["info"]["model_id"]
    loaded = smoke.client.get_logged_model(mid)
    document = smoke.document("logged_models", {"_id": mid})
    assert loaded.params == kv(document["params"]) == {"depth": "3"}
    assert loaded.tags == kv(document["tags"]) == {"team": "risk"}
    failure = smoke.http(
        "POST",
        "logged-models",
        {
            "experiment_id": eid,
            "name": "invalid",
            "tags": [{"key": "duplicate", "value": "a"}, {"key": "duplicate", "value": "b"}],
        },
        status=400,
    )
    assert failure["error_code"] == "BAD_REQUEST"
    assert smoke.client.search_logged_models([eid], filter_string="name = 'invalid'") == []
    assert smoke.documents("logged_models", {"name": "invalid"}) == []


def test_logged_model_requires_active_experiment(smoke):
    eid = smoke.experiment()
    smoke.client.delete_experiment(eid)
    assert (
        smoke.client.get_experiment(eid).lifecycle_stage
        == smoke.document("experiments", {"_id": eid})["lifecycle_stage"]
        == "deleted"
    )
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.create_logged_model(eid, name="forbidden")
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.create_logged_model("123456789", name="missing-parent")
    assert smoke.documents("logged_models", {}) == []


def test_logged_model_metrics_across_runs(smoke):
    eid = smoke.experiment()
    created = model(smoke, eid, name="multi-run")
    run_ids = []
    for index in range(2):
        rid = smoke.run(eid, name=f"run-{index}").info.run_id
        run_ids.append(rid)
        smoke.client.log_metric(
            rid,
            "loss",
            index + 0.5,
            timestamp=1000 + index,
            step=index,
            model_id=created.model_id,
            synchronous=True,
        )
        assert len(smoke.client.get_logged_model(created.model_id).metrics) == index + 1
        assert (
            smoke.document("run_metrics", {"run_id": rid, "model_id": created.model_id})["v"]
            == index + 0.5
        )
    assert {item.run_id for item in smoke.client.get_logged_model(created.model_id).metrics} == set(
        run_ids
    )


@pytest.fixture
def searchable_models(smoke):
    run = smoke.run()
    eid, rid = run.info.experiment_id, run.info.run_id
    models = {}
    for name, params, tags in [
        ("low", {"depth": "10"}, {"team.name": "Risk", "empty": ""}),
        ("high", {"depth": "2"}, {"team.name": "Platform", "empty": ""}),
        ("missing", {}, {}),
    ]:
        models[name] = model(smoke, eid, name=name, params=params, tags=tags)
    for name, value, timestamp, step, dataset, digest in [
        ("low", 0.05, 1000, 5, "eval", "a"),
        ("low", 0.8, 2000, 1, "train", "b"),
        ("high", 0.2, 1000, 1, "eval", "c"),
        ("high", 0.4, 2000, 1, "train", "b"),
    ]:
        mid = models[name].model_id
        smoke.client.log_batch(
            rid,
            metrics=[
                Metric(
                    "loss",
                    value,
                    timestamp,
                    step,
                    model_id=mid,
                    dataset_name=dataset,
                    dataset_digest=digest,
                )
            ],
            synchronous=True,
        )
        assert any(item.value == value for item in smoke.client.get_logged_model(mid).metrics)
        assert (
            smoke.document("run_metrics", {"model_id": mid, "timestamp": timestamp})["v"] == value
        )
    return eid, models


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("name = 'low'", {"low"}),
        ("params.depth = '10'", {"low"}),
        ("params.depth != '10'", {"high"}),
        ("params.depth IN ('10', '2')", {"low", "high"}),
        ("params.depth NOT IN ('10')", {"high"}),
        ("params.depth NOT IN ()", {"low", "high"}),
        ("params.depth LIKE '1%'", {"low"}),
        ("tags.`team.name` ILIKE 'risk'", {"low"}),
        ("tags.`team.name` LIKE 'risk'", set()),
        ("tags.empty = ''", {"low", "high"}),
        ("tags.empty NOT IN ()", {"low", "high"}),
        ("metrics.loss < 0.1", {"low"}),
        ("metrics.loss <= 0.2", {"low", "high"}),
        ("metrics.loss > 0.6", {"low"}),
        ("metrics.loss >= 0.8", {"low"}),
        ("metrics.loss = 0.4", {"high"}),
        ("metrics.loss != 0.4", {"low", "high"}),
        ("metrics.loss < 0.1 AND metrics.loss > 0.6", {"low"}),
        ("metrics.loss > 0.1 AND params.depth = '2'", {"high"}),
        ("tags.`team.name` = 'Risk' AND tags.`team.name` = 'Platform'", set()),
    ],
)
def test_logged_model_search_filters(smoke, searchable_models, expression, expected):
    eid, models = searchable_models
    before = smoke.documents("logged_models", {})
    found = pages(
        lambda token: smoke.client.search_logged_models(
            [eid],
            filter_string=expression,
            order_by=[{"field_name": "name", "ascending": True}],
            max_results=1,
            page_token=token,
        ),
        lambda item: item.model_id,
    )
    assert [item.name for item in found] == sorted(expected)
    assert {item.model_id for item in found} == {models[name].model_id for name in expected}
    assert smoke.documents("logged_models", {}) == before


@pytest.mark.parametrize(
    "datasets,expression,expected",
    [
        (
            [
                {"dataset_name": "eval", "dataset_digest": "a"},
                {"dataset_name": "eval", "dataset_digest": "c"},
            ],
            None,
            {"low", "high"},
        ),
        ([{"dataset_name": "eval", "dataset_digest": "a"}], None, {"low"}),
        ([{"dataset_name": "absent", "dataset_digest": "a"}], None, set()),
        (
            [
                {"dataset_name": "eval", "dataset_digest": "a"},
                {"dataset_name": "eval", "dataset_digest": "c"},
            ],
            "metrics.loss > 0.1",
            {"high"},
        ),
        ([{"dataset_name": "train", "dataset_digest": "b"}], "metrics.loss > 0.6", {"low"}),
    ],
)
def test_logged_model_dataset_search(smoke, searchable_models, datasets, expression, expected):
    eid, _ = searchable_models
    result = smoke.client.search_logged_models([eid], datasets=datasets, filter_string=expression)
    assert {item.name for item in result} == expected
    assert len(smoke.documents("logged_models", {"experiment_id": eid})) == 3


@pytest.mark.parametrize(
    "ascending,expected", [(True, ["high", "low", "missing"]), (False, ["low", "high", "missing"])]
)
def test_logged_model_metric_order(smoke, searchable_models, ascending, expected):
    eid, _ = searchable_models
    result = pages(
        lambda token: smoke.client.search_logged_models(
            [eid],
            order_by=[{"field_name": "metrics.loss", "ascending": ascending}],
            max_results=1,
            page_token=token,
        ),
        lambda item: item.model_id,
    )
    assert [item.name for item in result] == expected


def test_model_search_membership_scope_and_filter_sort_independence(smoke, searchable_models):
    eid, models = searchable_models
    low = models["low"].model_id
    assert [
        item.model_id
        for item in smoke.client.search_logged_models([eid], filter_string=f"model_id IN ('{low}')")
    ] == [low]
    result = smoke.client.search_logged_models(
        [eid],
        datasets=[
            {"dataset_name": "eval", "dataset_digest": "a"},
            {"dataset_name": "eval", "dataset_digest": "c"},
        ],
        filter_string="metrics.loss < 0.3",
        order_by=[
            {
                "field_name": "metrics.loss",
                "ascending": False,
                "dataset_name": "train",
                "dataset_digest": "b",
            }
        ],
    )
    assert [item.name for item in result] == ["low", "high"]
    assert smoke.client.search_logged_models([smoke.experiment("other")]) == []
    assert len(smoke.documents("logged_models", {"experiment_id": eid})) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"filter_string": "bad = 'x'"},
        {"page_token": "bad-token"},
        {"max_results": -1},
        {"order_by": [{"field_name": "bad"}]},
    ],
)
def test_invalid_logged_model_search(smoke, kwargs):
    eid = smoke.experiment()
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.search_logged_models([eid], **kwargs)


@pytest.mark.parametrize("alias", ["creation_time", "creation_timestamp", "last_updated_timestamp"])
def test_logged_model_timestamp_aliases(smoke, searchable_models, alias):
    eid, _ = searchable_models
    found = smoke.client.search_logged_models([eid], filter_string=f"{alias} > 0")
    assert {item.name for item in found} == {"low", "high", "missing"}
    assert len(smoke.documents("logged_models", {"experiment_id": eid})) == 3


@pytest.mark.skip(
    reason="Plugin finalize_logged_model is not implemented; flavors requiring it remain outside successful smoke coverage"
)
def test_logged_model_finalization_inventory():
    pass
