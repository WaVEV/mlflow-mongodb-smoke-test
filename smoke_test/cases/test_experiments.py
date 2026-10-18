import mlflow
from mlflow.entities import ViewType
from mlflow.utils.search_utils import SearchExperimentsUtils
import pytest

from smoke_test.support import error, kv, pages

pytestmark = pytest.mark.family("experiments")


def test_python_and_rest_creation(smoke):
    eid = smoke.experiment("python", {"team.name": "$risk", "empty": ""})
    exp = smoke.client.get_experiment_by_name("python")
    assert exp.experiment_id == eid
    assert exp.lifecycle_stage == "active"
    assert exp.creation_time <= exp.last_update_time
    location = (smoke.directory / "explicit-artifacts").as_uri()
    other = smoke.http(
        "POST",
        "experiments/create",
        {
            "name": "rest",
            "artifact_location": location,
            "tags": [{"key": "origin", "value": "REST"}],
        },
    )["experiment_id"]
    loaded = smoke.client.get_experiment(other)
    assert loaded.name == "rest" and loaded.artifact_location == location
    doc = smoke.document("experiments", {"_id": other})
    assert doc["artifact_location"] == location and kv(doc["tags"]) == {"origin": "REST"}


def test_fluent_selection_and_rename(smoke):
    selected = mlflow.set_experiment("fluent")
    eid = selected.experiment_id
    assert mlflow.set_experiment("fluent").experiment_id == eid
    before = smoke.document("experiments", {"_id": eid})
    assert smoke.client.get_experiment(eid).name == "fluent"
    assert len(smoke.documents("experiments", {"name": "fluent"})) == 1
    smoke.client.rename_experiment(eid, "renamed")
    assert (
        smoke.http("GET", "experiments/get-by-name", {"experiment_name": "renamed"})["experiment"][
            "experiment_id"
        ]
        == eid
    )
    assert smoke.client.get_experiment_by_name("fluent") is None
    after = smoke.document("experiments", {"_id": eid})
    assert after["name"] == "renamed" and after["artifact_location"] == before["artifact_location"]
    assert after["tags"] == before["tags"]
    smoke.client.rename_experiment(eid, "renamed")
    assert (
        smoke.client.get_experiment(eid).name == smoke.document("experiments", {"_id": eid})["name"]
    )


def assert_tag(smoke, eid, expected):
    loaded = smoke.http("GET", "experiments/get", {"experiment_id": eid})["experiment"]
    document = smoke.document("experiments", {"_id": eid})
    assert {tag["key"]: tag["value"] for tag in loaded.get("tags", [])} == expected
    assert kv(document["tags"]) == expected


def test_set_experiment_tag_is_required(smoke):
    eid = smoke.experiment()
    smoke.client.set_experiment_tag(eid, "smoke", "new")
    assert_tag(smoke, eid, {"keep": "yes", "smoke": "new"})
    assert [
        item.experiment_id
        for item in smoke.client.search_experiments(filter_string="tags.smoke = 'new'")
    ] == [eid]


def test_replace_experiment_tag_is_required(smoke):
    eid = smoke.experiment(tags={"keep": "yes", "smoke": "old"})
    smoke.client.set_experiment_tag(eid, "smoke", "$new")
    assert_tag(smoke, eid, {"keep": "yes", "smoke": "$new"})
    assert smoke.client.search_experiments(filter_string="tags.smoke = 'old'") == []
    assert [
        item.experiment_id
        for item in smoke.client.search_experiments(filter_string="tags.smoke = '$new'")
    ] == [eid]


def test_delete_experiment_tag_is_required(smoke):
    eid = smoke.experiment(tags={"keep": "yes", "smoke": "old"})
    smoke.client.delete_experiment_tag(eid, "smoke")
    assert_tag(smoke, eid, {"keep": "yes"})
    assert smoke.client.search_experiments(filter_string="tags.smoke = 'old'") == []


def test_delete_restore_and_views(smoke):
    eid = smoke.experiment()
    run = smoke.run(eid)
    smoke.client.delete_experiment(eid)
    assert smoke.client.get_experiment(eid).lifecycle_stage == "deleted"
    assert smoke.document("experiments", {"_id": eid})["lifecycle_stage"] == "deleted"
    assert smoke.client.get_run(run.info.run_id).info.lifecycle_stage == "active"
    assert smoke.document("runs", {"_id": run.info.run_id})["lifecycle_stage"] == "active"
    assert (
        smoke.client.search_experiments(ViewType.ACTIVE_ONLY, filter_string="name = 'smoke'") == []
    )
    assert [
        item.experiment_id for item in smoke.client.search_experiments(ViewType.DELETED_ONLY)
    ] == [eid]
    assert [
        item.experiment_id
        for item in smoke.client.search_experiments(ViewType.ALL, filter_string="name = 'smoke'")
    ] == [eid]
    # INVALID_STATE maps to HTTP 500; the SDK's retry adapter hides its JSON body.
    failure = smoke.http(
        "POST",
        "experiments/update",
        {
            "experiment_id": eid,
            "new_name": "invalid",
        },
        status=500,
    )
    assert failure["error_code"] == "INVALID_STATE"
    assert smoke.client.get_experiment(eid).name == "smoke"
    assert smoke.document("experiments", {"_id": eid})["name"] == "smoke"
    smoke.client.restore_experiment(eid)
    assert smoke.client.get_experiment(eid).lifecycle_stage == "active"
    assert smoke.document("experiments", {"_id": eid})["lifecycle_stage"] == "active"


@pytest.mark.parametrize("deleted", [False, True])
def test_duplicate_name(smoke, deleted):
    eid = smoke.experiment("duplicate")
    if deleted:
        smoke.client.delete_experiment(eid)
        assert smoke.client.get_experiment(eid).lifecycle_stage == "deleted"
        assert smoke.document("experiments", {"_id": eid})["lifecycle_stage"] == "deleted"
    with error("RESOURCE_ALREADY_EXISTS"):
        smoke.client.create_experiment("duplicate")
    assert len(smoke.documents("experiments", {"name": "duplicate"})) == 1


def test_missing_and_invalid_resources(smoke):
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_experiment("123456789")
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.create_experiment("")
    first = smoke.experiment("first")
    smoke.experiment("second")
    with error("RESOURCE_ALREADY_EXISTS"):
        smoke.client.rename_experiment(first, "second")
    assert (
        smoke.client.get_experiment(first).name
        == smoke.document("experiments", {"_id": first})["name"]
        == "first"
    )


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("name = 'RiskA'", {"RiskA"}),
        ("name != 'RiskA'", {"RiskB", "Other"}),
        ("name LIKE 'Risk%'", {"RiskA", "RiskB"}),
        ("name ILIKE 'risk_'", {"RiskA", "RiskB"}),
        ("name LIKE 'risk%'", set()),
        ("tags.team = 'risk'", {"RiskA"}),
        ("tags.team != 'risk'", {"RiskB"}),
        ("tags.team ILIKE 'RIS%'", {"RiskA"}),
        ("tags.team = 'risk' AND tags.purpose = 'risk'", set()),
        ("tags.team LIKE 'r%' AND tags.team != 'risk'", set()),
        ("tags.`team.name` = '$risk'", {"RiskA"}),
        ("tags.empty = ''", {"RiskA"}),
        ("creation_time > 0 AND last_update_time > 0", {"RiskA", "RiskB", "Other"}),
    ],
)
def test_experiment_search_filters(smoke, expression, expected):
    for name, tags in [
        ("RiskA", {"team": "risk", "purpose": "evaluation", "team.name": "$risk", "empty": ""}),
        ("RiskB", {"team": "platform", "purpose": "risk"}),
        ("Other", {"keep": "yes"}),
    ]:
        smoke.experiment(name, tags)
    before = smoke.documents("experiments", {})
    found = pages(
        lambda token: smoke.client.search_experiments(
            filter_string=expression, max_results=1, page_token=token, order_by=["name ASC"]
        ),
        lambda item: item.experiment_id,
    )
    assert {item.name for item in found} == expected
    assert [item.name for item in found] == sorted(expected)
    assert smoke.documents("experiments", {}) == before


@pytest.mark.parametrize(
    "direction,expected", [("ASC", ["a", "b", "c"]), ("DESC", ["c", "b", "a"])]
)
def test_experiment_ordering_pagination(smoke, direction, expected):
    for name in ("c", "a", "b"):
        smoke.experiment(name)
    found = pages(
        lambda token: smoke.client.search_experiments(
            max_results=1, page_token=token, order_by=[f"name {direction}"]
        ),
        lambda item: item.experiment_id,
    )
    assert [item.name for item in found] == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"filter_string": "unknown = 'x'"},
        {"page_token": "bad-token"},
        {"max_results": 0},
        {"order_by": ["unknown ASC"]},
    ],
)
def test_invalid_experiment_search(smoke, kwargs):
    smoke.experiment()
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.search_experiments(**kwargs)


@pytest.mark.parametrize(
    "comparator,expected",
    [
        pytest.param(
            operator,
            names,
            marks=[]
            if operator in SearchExperimentsUtils.VALID_TAG_COMPARATORS
            else [
                pytest.mark.upstream("Experiment tag existence comparator absent upstream"),
                pytest.mark.skip(reason="Experiment tag existence comparator absent upstream"),
            ],
        )
        for operator, names in [("IS NULL", {"missing"}), ("IS NOT NULL", {"empty", "present"})]
    ],
)
def test_experiment_tag_existence(smoke, comparator, expected):
    smoke.experiment("missing")
    smoke.experiment("empty", {"team": ""})
    smoke.experiment("present", {"team": "risk"})
    assert {
        item.name
        for item in smoke.client.search_experiments(filter_string=f"tags.team {comparator}")
    } == expected
    assert len(smoke.documents("experiments", {})) == 3
