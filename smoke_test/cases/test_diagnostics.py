from mlflow import MlflowClient
from mlflow.utils import search_utils
import pytest

from smoke_test.support import error

pytestmark = pytest.mark.family("diagnostics")


@pytest.mark.upstream("Evaluation dataset search unavailable in this upstream version")
@pytest.mark.skipif(
    not hasattr(MlflowClient, "search_datasets")
    or not hasattr(search_utils, "SearchEvaluationDatasetsUtils"),
    reason="No evaluation dataset search API upstream",
)
def test_evaluation_dataset_placeholder_contract(smoke):
    eid = smoke.experiment()
    result = smoke.client.search_datasets(experiment_ids=[eid])
    assert list(result) == [] and not result.token
    assert smoke.documents("runs", {}) == []
    with error("INVALID_PARAMETER_VALUE"):
        smoke.client.search_datasets(experiment_ids=[eid], filter_string="unknown = 'x'")


@pytest.mark.skip(
    reason="Advanced trace span/feedback/expectation/issue filters are ignored by the plugin; not supported searches"
)
def test_advanced_trace_filter_inventory():
    pass


@pytest.mark.skip(reason="Trace search model_id argument is not applied by the plugin")
def test_trace_model_filter_inventory():
    pass


@pytest.mark.skip(reason="unlink_traces_from_run is explicitly not implemented")
def test_trace_unlink_inventory():
    pass
