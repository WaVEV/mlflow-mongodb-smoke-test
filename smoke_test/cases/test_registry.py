from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
from mlflow.utils.search_utils import SearchModelVersionUtils
import pytest

from smoke_test.support import assert_tag_contents, error, pages

pytestmark = pytest.mark.family("registry")


def registered(smoke, name="smoke-model", tags=None):
    created = smoke.client.create_registered_model(
        name, tags=tags or {}, description="smoke description"
    )
    loaded = smoke.client.get_registered_model(name)
    document = smoke.document("registered_models", {"name": name})
    assert loaded.name == created.name == document["name"] == name
    assert loaded.description == document["description"] == "smoke description"
    assert loaded.tags == (tags or {})
    assert_tag_contents(document["tags"], tags or {})
    return document


def version(smoke, name="smoke-model", source="s3://smoke/metadata-only", **kwargs):
    created = smoke.client.create_model_version(
        name, source=source, description="version description", await_creation_for=0, **kwargs
    )
    loaded = smoke.client.get_model_version(name, created.version)
    parent = smoke.document("registered_models", {"name": name})
    document = smoke.document(
        "model_versions", {"registered_model_id": parent["_id"], "version": int(created.version)}
    )
    assert loaded.source == document["source"] == source
    assert loaded.status == document["status"] == "READY"
    assert loaded.description == document["description"] == "version description"
    return created


def test_registered_model_update_rename_and_alias_identity(smoke):
    original = registered(smoke)
    created = version(smoke)
    smoke.client.update_registered_model("smoke-model", "updated")
    assert (
        smoke.client.get_registered_model("smoke-model").description
        == smoke.document("registered_models", {"_id": original["_id"]})["description"]
        == "updated"
    )
    smoke.client.set_registered_model_alias("smoke-model", "candidate", created.version)
    assert (
        smoke.client.get_model_version_by_alias("smoke-model", "candidate").version
        == created.version
    )
    assert smoke.document("registered_models", {"_id": original["_id"]})["aliases"] == [
        {"alias": "candidate", "version": int(created.version)}
    ]
    smoke.client.rename_registered_model("smoke-model", "renamed")
    assert smoke.client.get_registered_model("renamed").aliases == {"candidate": created.version}
    assert smoke.document("registered_models", {"name": "renamed"})["_id"] == original["_id"]
    assert (
        smoke.client.get_model_version_by_alias("renamed", "candidate").version == created.version
    )
    assert smoke.document("model_versions", {"registered_model_id": original["_id"]})[
        "version"
    ] == int(created.version)
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_registered_model("smoke-model")


def test_registry_tag_mutations(smoke):
    parent = registered(smoke, tags={"keep": "yes"})
    created = version(smoke, tags={"keep": "yes"})
    for value in ("old", "$new"):
        smoke.client.set_registered_model_tag("smoke-model", "team.name", value)
        tags = smoke.document("registered_models", {"_id": parent["_id"]})["tags"]
        assert smoke.client.get_registered_model("smoke-model").tags["team.name"] == value
        assert_tag_contents(tags, {"keep": "yes", "team.name": value})
        smoke.client.set_model_version_tag("smoke-model", created.version, "team.name", value)
        tags = smoke.document(
            "model_versions",
            {"registered_model_id": parent["_id"], "version": int(created.version)},
        )["tags"]
        assert (
            smoke.client.get_model_version("smoke-model", created.version).tags["team.name"]
            == value
        )
        assert_tag_contents(tags, {"keep": "yes", "team.name": value})
    smoke.client.delete_registered_model_tag("smoke-model", "team.name")
    assert smoke.client.get_registered_model("smoke-model").tags == {"keep": "yes"}
    assert_tag_contents(
        smoke.document("registered_models", {"_id": parent["_id"]})["tags"], {"keep": "yes"}
    )
    smoke.client.delete_model_version_tag("smoke-model", created.version, "team.name")
    assert smoke.client.get_model_version("smoke-model", created.version).tags == {"keep": "yes"}
    assert_tag_contents(
        smoke.document(
            "model_versions",
            {"registered_model_id": parent["_id"], "version": int(created.version)},
        )["tags"],
        {"keep": "yes"},
    )


@pytest.mark.parametrize("collection", ["registered_models", "model_versions"])
def test_registry_persisted_tag_schema(smoke, collection):
    parent = registered(smoke, tags={"smoke": "value"})
    created = version(smoke, tags={"smoke": "value"})
    documents = {
        "registered_models": parent,
        "model_versions": smoke.document(
            "model_versions",
            {"registered_model_id": parent["_id"], "version": int(created.version)},
        ),
    }
    assert documents[collection]["tags"] == [{"key": "smoke", "value": "value"}]


def test_versions_stages_latest_and_alias_lifecycle(smoke):
    parent = registered(smoke)
    a, b = version(smoke), version(smoke)
    assert (a.version, b.version) == ("1", "2")
    assert smoke.client.get_model_version_by_alias("smoke-model", "latest").version == "2"
    assert smoke.document("registered_models", {"_id": parent["_id"]})["aliases"] == []
    smoke.client.update_model_version("smoke-model", a.version, "updated version")
    assert (
        smoke.client.get_model_version("smoke-model", a.version).description
        == smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})[
            "description"
        ]
        == "updated version"
    )
    smoke.client.transition_model_version_stage(
        "smoke-model", a.version, "Production", archive_existing_versions=False
    )
    assert (
        smoke.client.get_model_version("smoke-model", "1").current_stage
        == smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})[
            "current_stage"
        ]
        == "Production"
    )
    smoke.client.transition_model_version_stage(
        "smoke-model", b.version, "Production", archive_existing_versions=True
    )
    assert (
        smoke.client.get_model_version("smoke-model", "1").current_stage
        == smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})[
            "current_stage"
        ]
        == "Archived"
    )
    assert (
        smoke.client.get_model_version("smoke-model", "2").current_stage
        == smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 2})[
            "current_stage"
        ]
        == "Production"
    )
    assert [
        item.version
        for item in smoke.client.get_latest_versions("smoke-model", stages=["Production"])
    ] == ["2"]
    for target in ("1", "2"):
        smoke.client.set_registered_model_alias("smoke-model", "smoke", target)
        assert smoke.client.get_model_version_by_alias("smoke-model", "smoke").version == target
        assert smoke.document("registered_models", {"_id": parent["_id"]})["aliases"] == [
            {"alias": "smoke", "version": int(target)}
        ]
    smoke.client.delete_registered_model_alias("smoke-model", "smoke")
    assert smoke.client.get_registered_model("smoke-model").aliases == {}
    assert smoke.document("registered_models", {"_id": parent["_id"]})["aliases"] == []
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.set_registered_model_alias("smoke-model", "latest", "2")


def test_deleted_version_redaction_and_parent_cleanup(smoke):
    parent = registered(smoke)
    a, b = (
        version(smoke, run_id="smoke-run", run_link="https://example.invalid/smoke"),
        version(smoke),
    )
    smoke.client.set_registered_model_alias("smoke-model", "remove", a.version)
    assert smoke.client.get_model_version_by_alias("smoke-model", "remove").version == "1"
    assert smoke.document("registered_models", {"_id": parent["_id"]})["aliases"]
    smoke.client.delete_model_version("smoke-model", a.version)
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_model_version("smoke-model", a.version)
    doc = smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})
    assert doc["current_stage"] == "Deleted_Internal"
    assert (doc["source"], doc["run_id"], doc["run_link"], doc["description"]) == (
        "REDACTED-SOURCE-PATH",
        "REDACTED-RUN-ID",
        "REDACTED-RUN-LINK",
        None,
    )
    assert smoke.client.get_registered_model("smoke-model").aliases == {}
    assert smoke.document("registered_models", {"_id": parent["_id"]})["aliases"] == []
    assert [
        item.version for item in smoke.client.search_model_versions("name = 'smoke-model'")
    ] == [b.version]
    smoke.client.delete_registered_model("smoke-model")
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_registered_model("smoke-model")
    assert smoke.documents("model_versions", {"registered_model_id": parent["_id"]}) == []
    assert smoke.documents("registered_models", {"_id": parent["_id"]}) == []


def test_copy_version_and_real_local_artifacts(smoke):
    run = smoke.run()
    rid = run.info.run_id
    file = smoke.directory / "payload.txt"
    file.write_text("smoke artifact")
    smoke.client.log_artifact(rid, str(file), artifact_path="model")
    assert smoke.client.list_artifacts(rid, "model")[0].path == "model/payload.txt"
    assert smoke.document("runs", {"_id": rid})["artifact_uri"] == run.info.artifact_uri
    registered(smoke)
    source = f"runs:/{rid}/model"
    original = version(smoke, source=source, run_id=rid, tags={"copy": "yes"})
    copied = smoke.client.copy_model_version(
        f"models:/smoke-model/{original.version}", "smoke-copy"
    )
    loaded = smoke.client.get_model_version("smoke-copy", copied.version)
    parent = smoke.document("registered_models", {"name": "smoke-copy"})
    doc = smoke.document(
        "model_versions", {"registered_model_id": parent["_id"], "version": int(copied.version)}
    )
    assert loaded.source == doc["source"] == f"models:/smoke-model/{original.version}"
    assert loaded.run_id == doc["run_id"] == rid
    assert loaded.tags == {"copy": "yes"}
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/smoke-copy/{copied.version}/payload.txt",
        dst_path=str(smoke.directory / "download"),
    )
    assert Path(downloaded).read_text() == "smoke artifact"


def test_concurrent_version_allocation(smoke):
    parent = registered(smoke)

    def create(index):
        return smoke.client.create_model_version(
            "smoke-model", source=f"s3://smoke/{index}", await_creation_for=0
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        created = list(executor.map(create, range(3)))
    assert sorted(int(item.version) for item in created) == [1, 2, 3]
    for item in created:
        assert smoke.client.get_model_version("smoke-model", item.version).source == item.source
        assert (
            smoke.document(
                "model_versions",
                {"registered_model_id": parent["_id"], "version": int(item.version)},
            )["source"]
            == item.source
        )


@pytest.fixture
def searchable_registry(smoke):
    first = smoke.run(name="registry-one")
    second = smoke.run(first.info.experiment_id, name="registry-two")
    run_ids = {
        "run_one": first.info.run_id,
        "run_two": second.info.run_id,
        "run_two_upper": second.info.run_id.upper(),
    }
    for name, team in [("RiskA", "risk"), ("RiskB", "platform"), ("Other", "other")]:
        registered(smoke, name, tags={"team": team, "team.name": "$" + team})
        version(
            smoke,
            name,
            source="s3://smoke/A/B",
            run_id=run_ids["run_one"],
            tags={"quality": "good"},
        )
        version(
            smoke, name, source="s3://smoke/A/C", run_id=run_ids["run_two"], tags={"quality": "bad"}
        )
    return run_ids


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("name = 'RiskA'", {"RiskA"}),
        ("name LIKE 'Risk%'", {"RiskA", "RiskB"}),
        ("name ILIKE 'risk_'", {"RiskA", "RiskB"}),
        ("name LIKE 'risk%'", set()),
        ("tags.team = 'risk'", {"RiskA"}),
        ("tags.team != 'risk'", {"RiskB", "Other"}),
        ("tags.`team.name` = '$risk'", {"RiskA"}),
        ("tags.team = 'risk' AND tags.team = 'other'", set()),
    ],
)
def test_registered_model_search(smoke, searchable_registry, expression, expected):
    before = smoke.documents("registered_models", {})
    found = pages(
        lambda token: smoke.client.search_registered_models(
            filter_string=expression, order_by=["name ASC"], max_results=1, page_token=token
        ),
        lambda item: item.name,
    )
    assert [item.name for item in found] == sorted(expected)
    assert all(len(item.latest_versions) == 1 for item in found)
    assert smoke.documents("registered_models", {}) == before


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("name = 'RiskA'", {("RiskA", "1"), ("RiskA", "2")}),
        ("version_number = 2", {("RiskA", "2"), ("RiskB", "2"), ("Other", "2")}),
        ("version_number <= 1", {("RiskA", "1"), ("RiskB", "1"), ("Other", "1")}),
        ("run_id IN ('{run_one}')", {("RiskA", "1"), ("RiskB", "1"), ("Other", "1")}),
        ("run_id LIKE '{run_two}%'", {("RiskA", "2"), ("RiskB", "2"), ("Other", "2")}),
        ("run_id ILIKE '{run_two_upper}%'", {("RiskA", "2"), ("RiskB", "2"), ("Other", "2")}),
        ("source_path = 's3://smoke/A/C'", {("RiskA", "2"), ("RiskB", "2"), ("Other", "2")}),
        ("name = 'RiskA' AND tags.quality = 'good'", {("RiskA", "1")}),
        ("tags.quality = 'good' AND tags.quality = 'bad'", set()),
        pytest.param(
            "creation_timestamp > 0 AND last_updated_timestamp > 0",
            {
                ("RiskA", "1"),
                ("RiskA", "2"),
                ("RiskB", "1"),
                ("RiskB", "2"),
                ("Other", "1"),
                ("Other", "2"),
            },
            marks=[
                pytest.mark.upstream(
                    "Model-version timestamp filters absent from upstream validator"
                ),
                pytest.mark.skipif(
                    not {"creation_timestamp", "last_updated_timestamp"}.issubset(
                        SearchModelVersionUtils.VALID_SEARCH_ATTRIBUTE_KEYS
                    ),
                    reason="Model-version timestamp filters absent from upstream validator",
                ),
            ],
        ),
    ],
)
def test_model_version_search(smoke, searchable_registry, expression, expected):
    expression = expression.format(**searchable_registry)
    before = smoke.documents("model_versions", {})
    found = pages(
        lambda token: smoke.client.search_model_versions(
            filter_string=expression,
            order_by=["name ASC", "version_number ASC"],
            max_results=1,
            page_token=token,
        ),
        lambda item: (item.name, item.version),
    )
    assert {(item.name, item.version) for item in found} == expected
    assert [(item.name, item.version) for item in found] == sorted(expected)
    assert smoke.documents("model_versions", {}) == before


def test_registry_conflicts_and_missing(smoke):
    first = registered(smoke, "first")
    registered(smoke, "second")
    with error("RESOURCE_ALREADY_EXISTS"):
        smoke.client.create_registered_model("first")
    with error("RESOURCE_ALREADY_EXISTS"):
        smoke.client.rename_registered_model("first", "second")
    assert (
        smoke.client.get_registered_model("first").name
        == smoke.document("registered_models", {"_id": first["_id"]})["name"]
        == "first"
    )
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_registered_model("missing")
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_model_version("first", "99")


def test_registry_rest_creation(smoke):
    created = smoke.http(
        "POST",
        "registered-models/create",
        {"name": "rest-model", "description": "rest description"},
    )
    assert created["registered_model"]["name"] == "rest-model"
    parent = smoke.document("registered_models", {"name": "rest-model"})
    assert (
        smoke.client.get_registered_model("rest-model").description
        == parent["description"]
        == "rest description"
    )
    created = smoke.http(
        "POST",
        "model-versions/create",
        {"name": "rest-model", "source": "s3://smoke/metadata-only", "description": "rest version"},
    )
    number = created["model_version"]["version"]
    loaded = smoke.client.get_model_version("rest-model", str(number))
    assert (
        loaded.description
        == smoke.document(
            "model_versions", {"registered_model_id": parent["_id"], "version": int(number)}
        )["description"]
        == "rest version"
    )


@pytest.mark.upstream("Model-name IN filters absent from upstream parser")
@pytest.mark.skipif(
    "name" not in getattr(SearchModelVersionUtils, "LIST_SUPPORTED_KEYS", ()),
    reason="No model-name IN parser upstream",
)
def test_model_version_name_membership(smoke, searchable_registry):
    result = smoke.client.search_model_versions("name IN ('RiskA', 'RiskB')")
    assert {(item.name, item.version) for item in result} == {
        ("RiskA", "1"),
        ("RiskA", "2"),
        ("RiskB", "1"),
        ("RiskB", "2"),
    }
    assert len(smoke.documents("model_versions", {})) == 6


@pytest.mark.parametrize(
    "direction,expected",
    [("ASC", ["Other", "RiskA", "RiskB"]), ("DESC", ["RiskB", "RiskA", "Other"])],
)
def test_registered_model_ordering(smoke, searchable_registry, direction, expected):
    result = pages(
        lambda token: smoke.client.search_registered_models(
            order_by=[f"name {direction}"], max_results=1, page_token=token
        ),
        lambda item: item.name,
    )
    assert [item.name for item in result] == expected


@pytest.mark.parametrize(
    "kwargs", [{"filter_string": "unknown = 'x'"}, {"page_token": "bad-token"}, {"max_results": 0}]
)
def test_invalid_registered_model_search(smoke, kwargs):
    registered(smoke)
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.search_registered_models(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"filter_string": "unknown = 'x'"}, {"page_token": "bad-token"}, {"max_results": 0}]
)
def test_invalid_model_version_search(smoke, kwargs):
    registered(smoke)
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.search_model_versions(**kwargs)


def test_stage_transition_without_archiving_preserves_peers(smoke):
    parent = registered(smoke)
    first, second = version(smoke), version(smoke)
    for current in (first, second):
        smoke.client.transition_model_version_stage(
            "smoke-model", current.version, "Production", archive_existing_versions=False
        )
        assert (
            smoke.client.get_model_version("smoke-model", current.version).current_stage
            == "Production"
        )
        assert (
            smoke.document(
                "model_versions",
                {"registered_model_id": parent["_id"], "version": int(current.version)},
            )["current_stage"]
            == "Production"
        )
    assert (
        smoke.client.get_model_version("smoke-model", first.version).current_stage == "Production"
    )
    assert (
        smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})[
            "current_stage"
        ]
        == "Production"
    )
    assert [
        item.version for item in smoke.client.get_latest_versions("smoke-model", ["Production"])
    ] == ["2"]
