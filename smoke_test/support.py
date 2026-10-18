"""HTTP, Mongo evidence, and disposable MLflow server lifecycle."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import mlflow
from bson import json_util
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds
from pymongo import MongoClient
from pymongo.uri_parser import parse_uri
import requests


def database_uri(uri, name):
    parsed = urlsplit(uri)
    if parsed.scheme not in ("mongodb", "mongodb+srv"):
        raise ValueError("MONGODB_URI must use mongodb:// or mongodb+srv://")
    options = parse_qsl(parsed.query, keep_blank_values=True)
    if "@" in parsed.netloc and not any(key.lower() == "authsource" for key, _ in options):
        # Replacing the database must not silently change password authentication's DB.
        auth_source = unquote(parsed.path.lstrip("/")) or "admin"
        if parsed.scheme == "mongodb+srv":
            # A TXT record may specify authSource; do not override that implicit choice.
            resolved = parse_uri(uri, connect_timeout=5)
            auth_source = resolved["options"].get("authSource", auth_source)
        options.append(("authSource", auth_source))
    return urlunsplit((parsed.scheme, parsed.netloc, "/" + name, urlencode(options), ""))


def kv(entries):
    assert all(
        "k" in item and "v" in item and "key" not in item and "value" not in item
        for item in entries
    ), entries
    assert len({item["k"] for item in entries}) == len(entries), entries
    return {item["k"]: item["v"] for item in entries}


def assert_tag_contents(entries, expected):
    """Check persisted tag data independently of the separate mandatory schema scenarios."""
    assert len(entries) == len(expected), entries
    assert sorted(sorted(entry.values()) for entry in entries) == sorted(
        sorted((key, value)) for key, value in expected.items()
    ), entries


def assert_native_value(actual, expected):
    if type(expected) in (int, float):
        # Protobuf Value transports both Python numeric types as doubles.
        assert type(actual) in (int, float)
    else:
        assert type(actual) is type(expected)
    assert actual == expected
    if isinstance(expected, list):
        for actual_item, expected_item in zip(actual, expected):
            assert_native_value(actual_item, expected_item)
    elif isinstance(expected, dict):
        for key, expected_item in expected.items():
            assert_native_value(actual[key], expected_item)


def pages(fetch, identity):
    values, tokens = [], set()
    token = None
    for _ in range(100):
        page = fetch(token)
        values.extend(page)
        token = getattr(page, "token", None)
        if not token:
            ids = [identity(value) for value in values]
            assert len(ids) == len(set(ids)), f"Duplicate results: {ids}"
            return values
        assert token not in tokens, "Search repeated a page token"
        tokens.add(token)
    raise AssertionError("Search did not terminate after 100 pages")


@contextmanager
def error(code):
    try:
        yield
    except MlflowException as exc:
        assert exc.error_code == code, f"Expected {code}, received {exc.error_code}: {exc}"
    else:
        raise AssertionError(f"Expected MLflow error {code}")


class Scenario:
    def __init__(self, nodeid):
        self.database_name = "mlflow_smoke_" + uuid4().hex
        self.directory = (
            Path(os.environ["SMOKE_OUTPUT"]) / hashlib.sha256(nodeid.encode()).hexdigest()[:16]
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "scenario.json").write_text(
            json.dumps(
                {
                    "case": nodeid,
                    "database": self.database_name,
                    "mlflow": mlflow.__version__,
                    "python": sys.version,
                    "artifacts": str(self.directory / "artifacts"),
                },
                indent=2,
            )
        )
        self.uri = database_uri(os.environ["MONGODB_URI"], self.database_name)
        self.mongo = None
        self.process = None
        self.reader = None
        self.owns_database = False
        self.port = self.free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.client = MlflowClient(tracking_uri=self.url, registry_uri=self.url)
        # This is MLflow's HTTP transport, never the plugin store instance.
        self.rest = RestStore(lambda: MlflowHostCreds(host=self.url))
        self.lock = threading.Lock()

    @staticmethod
    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def record(self, kind, **values):
        with self.lock, (self.directory / "evidence.jsonl").open("a") as log:
            log.write(json_util.dumps({"time": time.time(), "kind": kind, **values}) + "\n")

    def start(self):
        self.mongo = MongoClient(self.uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=10000)
        try:
            self.mongo.admin.command("ping")
            version = self.mongo.server_info()["versionArray"]
            assert tuple(version[:2]) >= (8, 0), "MongoDB 8.0+ required"
            assert self.database_name not in self.mongo.list_database_names(), (
                "Generated DB already exists"
            )
        except Exception:
            raise RuntimeError(
                "Cannot initialize dedicated MongoDB database; check connection, version and listDatabases permissions"
            ) from None
        self.owns_database = True
        self.db = self.mongo[self.database_name]
        self.record("environment", mongodb_version=version, database=self.database_name)
        self.start_server()
        mlflow.set_tracking_uri(self.url)
        mlflow.set_registry_uri(self.url)

    def start_server(self):
        environment = os.environ.copy()
        for key in list(environment):
            if key.startswith("MLFLOW_"):
                environment.pop(key)
        environment.update(
            {
                "MLFLOW_BACKEND_STORE_URI": self.uri,
                "MLFLOW_REGISTRY_STORE_URI": self.uri,
                "MLFLOW_HTTP_REQUEST_TIMEOUT": "15",
                "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
                "MLFLOW_ENABLE_TELEMETRY": "false",
                "DO_NOT_TRACK": "1",
            }
        )
        argv = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--workers",
            "1",
            "--no-serve-artifacts",
            "--default-artifact-root",
            (self.directory / "artifacts").as_uri(),
        ]
        self.process = subprocess.Popen(
            argv,
            cwd=self.directory,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        (self.directory / "server.json").write_text(
            json.dumps(
                {
                    "pid": self.process.pid,
                    "url": self.url,
                    "database": self.database_name,
                },
                indent=2,
            )
            + "\n"
        )

        def drain():
            with (self.directory / "server.log").open("a") as log:
                for line in self.process.stdout:
                    log.write(line.replace(self.uri, "<dedicated MongoDB URI>"))
                    log.flush()

        self.reader = threading.Thread(target=drain, daemon=True)
        self.reader.start()
        deadline = time.monotonic() + int(os.environ.get("SMOKE_STARTUP_TIMEOUT", "90"))
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("MLflow server exited during startup; see server.log")
            try:
                response = requests.get(self.url + "/health", timeout=1)
                if response.status_code == 200:
                    version = requests.get(self.url + "/version", timeout=5)
                    assert (
                        version.status_code == 200 and version.text.strip() == mlflow.__version__
                    ), "Client/server MLflow versions differ"
                    self.record("server.ready", version=version.text.strip(), url=self.url)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise RuntimeError("Timed out waiting for MLflow server; see server.log")

    def stop_server(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
            self.reader.join(timeout=5)
            self.process.stdout.close()
            self.record("server.stopped", pid=self.process.pid)
            self.process = None

    def restart(self):
        self.stop_server()
        self.start_server()

    def close(self, keep=False):
        try:
            self.stop_server()
        finally:
            if self.mongo is not None:
                try:
                    if self.owns_database and not keep:
                        assert self.database_name.startswith("mlflow_smoke_")
                        self.mongo.drop_database(self.database_name)
                    self.record("cleanup", database=self.database_name, preserved=keep)
                finally:
                    self.mongo.close()

    def http(self, method, path, payload=None, status=200):
        response = requests.request(
            method,
            self.url + "/api/2.0/mlflow/" + path,
            params=payload if method == "GET" else None,
            json=payload if method != "GET" else None,
            timeout=20,
        )
        assert response.status_code == status, (
            f"{method} {path}: {response.status_code}: {response.text}"
        )
        return response.json()

    def document(self, collection, query):
        result = self.db[collection].find_one(query)
        self.record(
            "mongo.find_one",
            collection=collection,
            query=query,
            result=result,
            mongosh=self.mongo_query(collection, query, "findOne"),
        )
        assert result is not None, f"Missing Mongo document: {collection} {query}"
        return result

    def documents(self, collection, query):
        with self.db[collection].find(query) as cursor:
            result = list(cursor)
        self.record(
            "mongo.find",
            collection=collection,
            query=query,
            result=result,
            mongosh=self.mongo_query(collection, query, "find"),
        )
        return result

    def mongo_query(self, collection, query, operation):
        encoded = json_util.dumps(query, json_options=json_util.CANONICAL_JSON_OPTIONS)
        return (
            f"db.getSiblingDB({json.dumps(self.database_name)}).getCollection({json.dumps(collection)})"
            f".{operation}(EJSON.deserialize({encoded}))"
        )

    def experiment(self, name="smoke", tags=None):
        eid = self.client.create_experiment(name, tags=tags or {"keep": "yes"})
        loaded = self.http("GET", "experiments/get", {"experiment_id": eid})["experiment"]
        assert loaded["name"] == name
        document = self.document("experiments", {"_id": eid})
        assert document["name"] == name
        assert kv(document["tags"]) == (tags or {"keep": "yes"})
        return eid

    def run(self, experiment_id=None, name="smoke-run", **kwargs):
        eid = experiment_id or self.experiment()
        run = self.client.create_run(eid, run_name=name, **kwargs)
        rid = run.info.run_id
        assert self.http("GET", "runs/get", {"run_id": rid})["run"]["info"]["experiment_id"] == eid
        assert self.document("runs", {"_id": rid})["experiment_id"] == eid
        return run

    def trace(self, experiment_id=None, name="smoke-trace", attributes=None):
        eid = experiment_id or self.experiment()
        mlflow.set_experiment(experiment_id=eid)
        with mlflow.start_span(name=name) as span:
            span.set_inputs({"question": "2 + 2"})
            span.set_outputs({"answer": 4})
            span.set_attributes(attributes or {"provider": "smoke", "temperature": 0.2})
            trace_id = span.trace_id
        mlflow.flush_trace_async_logging()
        trace = self.wait_trace(trace_id)
        assert trace.info.trace_id == trace_id
        assert trace.data.spans[0].inputs == {"question": "2 + 2"}
        assert self.document("traces", {"_id": trace_id})["experiment_id"] == eid
        from mlflow.entities.span import Span

        stored = self.documents("spans", {"trace_id": trace_id})
        assert len(stored) == 1
        payload = Span.from_dict(stored[0]["content"])
        assert payload.inputs == {"question": "2 + 2"} and payload.outputs == {"answer": 4}
        return trace

    def wait_trace(self, trace_id, expected_spans=1):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                trace = self.client.get_trace(trace_id)
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
            else:
                stored = self.documents("spans", {"trace_id": trace_id})
                if len(trace.data.spans) == len(stored) == expected_spans:
                    return trace
            time.sleep(0.2)
        raise AssertionError(
            f"Trace {trace_id}: full HTTP payload and {expected_spans} Mongo spans were not available within 30s"
        )

    def search_traces(self, eid, **kwargs):
        # experiment_ids exists throughout the supported range (locations was added later).
        return self.client.search_traces(experiment_ids=[eid], **kwargs)

    def trace_info(self, experiment_id, **overrides):
        from mlflow.entities import TraceInfo, TraceLocation, TraceState

        values = dict(
            trace_id="tr-" + uuid4().hex,
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=1700000000000,
            execution_duration=10,
            state=TraceState.OK,
        )
        values.update(overrides)
        info = self.rest.start_trace(TraceInfo(**values))
        assert self.rest.get_trace_info(info.trace_id).trace_id == info.trace_id
        assert self.document("traces", {"_id": info.trace_id})["experiment_id"] == experiment_id
        return info

    def fresh_read(self, code):
        environment = os.environ.copy() | {
            "MLFLOW_TRACKING_URI": self.url,
            "MLFLOW_REGISTRY_URI": self.url,
        }
        # A fresh scenario must not inherit a run/experiment from a prior database.
        for key in ("MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_RUN_ID"):
            environment.pop(key, None)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.record(
            "fresh_client",
            code=code,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
        assert completed.returncode == 0, completed.stderr


def span(
    trace_id, number=1, parent=None, start=1700000000000000000, duration=10000000, attributes=None
):
    from mlflow.entities.span import Span
    from mlflow.tracing.constant import SpanAttributeKey
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import SpanContext, Status, StatusCode

    context_trace = int(trace_id.removeprefix("tr-"), 16)
    values = {SpanAttributeKey.REQUEST_ID: trace_id, SpanAttributeKey.SPAN_TYPE: "CHAIN"}
    values.update(attributes or {})
    return Span(
        ReadableSpan(
            name=f"span-{number}",
            context=SpanContext(trace_id=context_trace, span_id=number, is_remote=False),
            parent=SpanContext(trace_id=context_trace, span_id=parent, is_remote=False)
            if parent
            else None,
            start_time=start,
            end_time=start + duration,
            status=Status(StatusCode.OK),
            attributes=values,
            resource=Resource({}),
        )
    )
