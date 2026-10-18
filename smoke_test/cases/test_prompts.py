import mlflow
from mlflow import MlflowClient
from mlflow.prompt.constants import PROMPT_TEXT_TAG_KEY
from packaging.version import Version
import pytest

from smoke_test.support import assert_tag_contents, error, pages

pytestmark = pytest.mark.family("prompts")


def register(smoke, name="smoke-prompt", template="Summarize {{document}}.", tags=None):
    created = smoke.client.register_prompt(
        name, template, commit_message="smoke commit", tags=tags or {"owner": "smoke"}
    )
    loaded = smoke.client.get_prompt_version(name, created.version)
    assert loaded.template == template and loaded.commit_message == "smoke commit"
    parent = smoke.document("registered_models", {"name": name})
    document = smoke.document(
        "model_versions", {"registered_model_id": parent["_id"], "version": int(created.version)}
    )
    template_tags = [entry for entry in document["tags"] if PROMPT_TEXT_TAG_KEY in entry.values()]
    assert_tag_contents(template_tags, {PROMPT_TEXT_TAG_KEY: template})
    return created


@pytest.mark.upstream("create_prompt metadata API not exposed by this MLflow client")
@pytest.mark.skipif(
    not hasattr(MlflowClient, "create_prompt"), reason="create_prompt absent upstream"
)
def test_prompt_metadata_tags(smoke):
    smoke.client.create_prompt("smoke-prompt", description="description", tags={"keep": "yes"})
    parent = smoke.document("registered_models", {"name": "smoke-prompt"})
    assert (
        smoke.client.get_prompt("smoke-prompt").description
        == parent["description"]
        == "description"
    )
    for value in ("old", "$new"):
        smoke.client.set_prompt_tag("smoke-prompt", "team.name", value)
        assert smoke.client.get_prompt("smoke-prompt").tags["team.name"] == value
        persisted = smoke.document("registered_models", {"_id": parent["_id"]})["tags"]
        assert_tag_contents(
            persisted, {"keep": "yes", "team.name": value, "mlflow.prompt.is_prompt": "true"}
        )
    smoke.client.delete_prompt_tag("smoke-prompt", "team.name")
    assert smoke.client.get_prompt("smoke-prompt").tags == {"keep": "yes"}
    assert_tag_contents(
        smoke.document("registered_models", {"_id": parent["_id"]})["tags"],
        {"keep": "yes", "mlflow.prompt.is_prompt": "true"},
    )


def test_prompt_versions_tags_aliases_render_and_delete(smoke):
    first = register(smoke)
    second = register(smoke, template="Explain {{document}}.")
    assert (int(first.version), int(second.version)) == (1, 2)
    parent = smoke.document("registered_models", {"name": first.name})
    smoke.client.set_prompt_version_tag(first.name, first.version, "reviewed", "$yes")
    assert smoke.client.get_prompt_version(first.name, 1).tags["reviewed"] == "$yes"
    tags = smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})[
        "tags"
    ]
    assert any("$yes" in item.values() and "reviewed" in item.values() for item in tags)
    smoke.client.delete_prompt_version_tag(first.name, first.version, "reviewed")
    assert "reviewed" not in smoke.client.get_prompt_version(first.name, 1).tags
    tags = smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})[
        "tags"
    ]
    assert all("reviewed" not in item.values() for item in tags)
    for target in (1, 2):
        smoke.client.set_prompt_alias(first.name, "smoke", target)
        assert smoke.client.get_prompt_version(first.name, "smoke").version == target
        assert smoke.document("registered_models", {"_id": parent["_id"]})["aliases"] == [
            {"alias": "smoke", "version": target}
        ]
    smoke.fresh_read(
        "from mlflow import MlflowClient; "
        "p=MlflowClient().load_prompt('prompts:/smoke-prompt@smoke'); "
        "assert p.format(document='hello')=='Explain hello.'"
    )
    smoke.client.delete_prompt_alias(first.name, "smoke")
    assert smoke.document("registered_models", {"_id": parent["_id"]})["aliases"] == []
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.get_prompt_version(first.name, "smoke")
    smoke.client.delete_prompt_version(first.name, 1)
    with error("RESOURCE_DOES_NOT_EXIST"):
        smoke.client.get_prompt_version(first.name, 1)
    assert (
        smoke.document("model_versions", {"registered_model_id": parent["_id"], "version": 1})[
            "current_stage"
        ]
        == "Deleted_Internal"
    )
    assert smoke.client.get_prompt_version(first.name, 2).template == second.template
    smoke.client.delete_prompt(first.name)
    assert smoke.client.get_prompt(first.name) is None
    assert smoke.documents("registered_models", {"_id": parent["_id"]}) == []
    assert smoke.documents("model_versions", {"registered_model_id": parent["_id"]}) == []


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("name LIKE 'smoke-%'", {"smoke-a", "smoke-b"}),
        ("name = 'smoke-a'", {"smoke-a"}),
        ("tags.owner = 'a'", {"smoke-a"}),
        ("tags.owner = 'absent'", set()),
    ],
)
def test_prompt_search_and_model_separation(smoke, expression, expected):
    for name, owner in [("smoke-a", "a"), ("smoke-b", "b")]:
        register(smoke, name, tags={"owner": owner})
    smoke.client.create_registered_model("smoke-model")
    assert (
        smoke.client.get_registered_model("smoke-model").name
        == smoke.document("registered_models", {"name": "smoke-model"})["name"]
    )
    found = pages(
        lambda token: smoke.client.search_prompts(
            filter_string=expression, max_results=1, page_token=token
        ),
        lambda item: item.name,
    )
    assert sorted(item.name for item in found) == sorted(expected)
    assert [item.name for item in smoke.client.search_registered_models()] == ["smoke-model"]
    assert len(smoke.documents("registered_models", {})) == 3


@pytest.mark.upstream("OSS prompt-version search requires MLflow >=3.11.1")
@pytest.mark.skipif(
    Version(mlflow.__version__) < Version("3.11.1"), reason="OSS prompt version search unavailable"
)
def test_prompt_version_search(smoke):
    for index in range(3):
        register(smoke, template=f"Version {index}: {{{{document}}}}")
    found = pages(
        lambda token: smoke.client.search_prompt_versions(
            "smoke-prompt", max_results=1, page_token=token
        ),
        lambda item: item.version,
    )
    assert [item.version for item in found] == [3, 2, 1]
    assert len(smoke.documents("model_versions", {})) == 3


def test_prompt_model_name_conflicts(smoke):
    register(smoke)
    with error("RESOURCE_ALREADY_EXISTS"):
        smoke.client.create_registered_model("smoke-prompt")
    assert smoke.client.get_prompt_version("smoke-prompt", 1).template == "Summarize {{document}}."
    assert len(smoke.documents("registered_models", {"name": "smoke-prompt"})) == 1
