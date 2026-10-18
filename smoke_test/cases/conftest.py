"""Isolated real-server fixtures; no direct calls to MongoDB plugin stores."""

import json
import inspect
import os
from pathlib import Path

import pytest

from smoke_test.support import Scenario

REPORTS = {}


def save_reports():
    output = Path(os.environ["SMOKE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "cases.json.tmp"
    temporary.write_text(json.dumps(list(REPORTS.values()), indent=2) + "\n")
    temporary.replace(output / "cases.json")


def pytest_configure(config):
    config.addinivalue_line("markers", "family(name): smoke family")
    config.addinivalue_line("markers", "upstream(reason): unavailable upstream capability")
    from mlflow import MlflowClient
    from mlflow.store.tracking.rest_store import RestStore
    from mlflow.protos import model_registry_pb2, service_pb2
    from mlflow.utils.rest_utils import extract_api_info_for_service

    output = Path(os.environ["SMOKE_OUTPUT"])
    methods = {}
    for cls in (MlflowClient, RestStore):
        methods[cls.__name__] = {
            name: str(inspect.signature(method))
            for name, method in inspect.getmembers(cls, inspect.isfunction)
            if not name.startswith("_")
        }
    routes = extract_api_info_for_service(service_pb2.MlflowService, "/api/2.0")
    registry_routes = extract_api_info_for_service(
        model_registry_pb2.ModelRegistryService, "/api/2.0"
    )
    (output / "capabilities.json").write_text(
        json.dumps(
            {
                "methods": methods,
                "tracking_routes": {method.__name__: value for method, value in routes.items()},
                "registry_routes": {
                    method.__name__: value for method, value in registry_routes.items()
                },
                "route_note": "Declared protobuf routes; version-specific overrides and actual requests are in evidence.jsonl",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


def pytest_collection_modifyitems(config, items):
    families = set(filter(None, os.environ.get("SMOKE_FAMILIES", "").split(",")))
    if families:
        selected, deselected = [], []
        for item in items:
            marker = item.get_closest_marker("family")
            (selected if marker and marker.args[0] in families else deselected).append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    entry = REPORTS.setdefault(item.nodeid, {"id": item.nodeid, "status": "BLOCKED", "phases": []})
    entry["phases"].append(
        {"phase": report.when, "outcome": report.outcome, "seconds": report.duration}
    )
    if report.failed:
        entry["status"] = "SETUP_FAILED" if report.when == "setup" else "FAILED"
        entry["error"] = str(report.longrepr)
        item.smoke_failed = True
    elif report.skipped:
        entry["status"] = (
            "UNAVAILABLE_UPSTREAM" if item.get_closest_marker("upstream") else "NOT_IMPLEMENTED"
        )
        entry["reason"] = str(report.longrepr)
    elif report.when == "call" and entry["status"] == "BLOCKED":
        entry["status"] = "PASSED"
    scenario = getattr(item, "smoke_scenario", None)
    if scenario is not None:
        entry["database"] = scenario.database_name
        entry["evidence"] = str(scenario.directory)
    save_reports()


def pytest_sessionfinish(session, exitstatus):
    save_reports()


@pytest.fixture
def smoke(request, monkeypatch):
    scenario = Scenario(request.node.nodeid)
    request.node.smoke_scenario = scenario
    # Audit only local smoke traffic. Never persist authentication headers or Mongo URIs.
    import requests

    original = requests.sessions.Session.send

    def send(session, prepared, **kwargs):
        try:
            response = original(session, prepared, **kwargs)
        except requests.RequestException as exc:
            if prepared.url.startswith(scenario.url + "/"):
                scenario.record(
                    "http.failed",
                    method=prepared.method,
                    url=prepared.url,
                    error=type(exc).__name__,
                )
            raise
        if prepared.url.startswith(scenario.url + "/"):
            scenario.record(
                "http",
                method=prepared.method,
                url=prepared.url,
                request=prepared.body,
                status=response.status_code,
                response=response.text[:32000],
            )
        return response

    monkeypatch.setattr(requests.sessions.Session, "send", send)
    ready = False
    try:
        scenario.start()
        ready = True
        scenario.record("scenario.start", operation=request.node.name)
        yield scenario
    finally:
        failed = not ready or getattr(request.node, "smoke_failed", False)
        scenario.record("scenario.finish", failed=failed)
        scenario.close(keep=failed and os.environ.get("SMOKE_KEEP_FAILED_DB") == "1")
