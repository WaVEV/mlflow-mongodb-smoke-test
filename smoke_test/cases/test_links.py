import json

from mlflow import MlflowClient
from mlflow.utils.search_utils import SearchTraceUtils
import pytest

from smoke_test.support import kv

pytestmark = pytest.mark.family("links")


@pytest.mark.upstream("Prompt/run linking method absent in upstream client")
@pytest.mark.skipif(
    not hasattr(MlflowClient, "link_prompt_version_to_run"),
    reason="No public prompt/run linking API",
)
def test_prompt_run_link(smoke):
    from smoke_test.cases.test_prompts import register

    run = smoke.run()
    prompt = register(smoke)
    expected = [{"name": prompt.name, "version": str(prompt.version)}]
    for _ in range(2):
        smoke.client.link_prompt_version_to_run(run.info.run_id, prompt)
        tag = smoke.client.get_run(run.info.run_id).data.tags["mlflow.linkedPrompts"]
        assert json.loads(tag) == expected
        assert (
            json.loads(
                kv(smoke.document("runs", {"_id": run.info.run_id})["tags"])["mlflow.linkedPrompts"]
            )
            == expected
        )


@pytest.mark.upstream("Prompt/model linking method absent in upstream client")
@pytest.mark.skipif(
    not hasattr(MlflowClient, "link_prompt_version_to_model"),
    reason="No public prompt/model linking API",
)
def test_prompt_logged_model_link(smoke):
    from smoke_test.cases.test_prompts import register
    from smoke_test.cases.test_logged_models import model

    eid = smoke.experiment()
    created = model(smoke, eid, name="linked")
    prompt = register(smoke)
    expected = [{"name": prompt.name, "version": str(prompt.version)}]
    for _ in range(2):
        smoke.client.link_prompt_version_to_model(
            prompt.name, str(prompt.version), created.model_id
        )
        tag = smoke.client.get_logged_model(created.model_id).tags["mlflow.linkedPrompts"]
        assert json.loads(tag) == expected
        assert (
            json.loads(
                kv(smoke.document("logged_models", {"_id": created.model_id})["tags"])[
                    "mlflow.linkedPrompts"
                ]
            )
            == expected
        )


@pytest.mark.upstream("Prompt/trace linking method absent in upstream client")
@pytest.mark.skipif(
    not hasattr(MlflowClient, "link_prompt_versions_to_trace"),
    reason="No public prompt/trace linking API",
)
def test_prompt_trace_link_and_search(smoke):
    from smoke_test.cases.test_prompts import register

    trace = smoke.trace()
    prompt = register(smoke)
    expected = [{"name": prompt.name, "version": str(prompt.version)}]
    for _ in range(2):
        smoke.client.link_prompt_versions_to_trace([prompt], trace.info.trace_id)
        assert (
            json.loads(smoke.rest.get_trace_info(trace.info.trace_id).tags["mlflow.linkedPrompts"])
            == expected
        )
        assert (
            json.loads(
                kv(smoke.document("traces", {"_id": trace.info.trace_id})["tags"])[
                    "mlflow.linkedPrompts"
                ]
            )
            == expected
        )
    if "prompt" in SearchTraceUtils.SEARCH_KEY_TO_TAG:
        found = smoke.search_traces(
            trace.info.experiment_id,
            filter_string=f"prompt = '{prompt.name}/{prompt.version}'",
            include_spans=False,
        )
        assert [item.info.trace_id for item in found] == [trace.info.trace_id]


@pytest.mark.upstream("Explicit trace/run linking method absent in upstream client")
@pytest.mark.skipif(
    not hasattr(MlflowClient, "link_traces_to_run"), reason="No public trace/run linking API"
)
def test_trace_run_links_and_search(smoke):
    run = smoke.run()
    trace = smoke.trace(run.info.experiment_id)
    rid, tid = run.info.run_id, trace.info.trace_id
    for _ in range(2):
        smoke.client.link_traces_to_run([tid], rid)
        found = smoke.search_traces(run.info.experiment_id, run_id=rid, include_spans=False)
        assert [item.info.trace_id for item in found] == [tid]
        assert smoke.document("traces", {"_id": tid})["run_ids"] == [rid]
