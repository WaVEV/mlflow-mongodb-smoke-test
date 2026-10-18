# MLflow smoke tests over HTTP and MongoDB

The runner installs the plugin checkout specified by `--plugin-dir` in a temporary environment for each MLflow version.
It starts its own server on an available port and uses a new MongoDB database for each scenario.
It does not use the server at `localhost:5000` or delete the database specified in your URI.

You need an accessible MongoDB 8.0+ instance and Python 3.10+ with `venv` and access to PyPI
or your package mirror. Python 3.11 or 3.12 is recommended for running older and newer releases
with the same interpreter. Docker is optional: an existing MongoDB instance works.

Clone this repository and the plugin separately. From this repository's root, specify the
path to the plugin checkout you want to test. The checkout may contain uncommitted changes.
Start with one version:

```bash
export MONGODB_URI='mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000'
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1
```

Then run the full matrix of stable 3.x versions starting with 3.5, including patch releases:

```bash
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions-all
```

You do not need to activate the project's `.venv` or install pytest in the runner's environment.
The runner uses the standard library and installs MLflow, the plugin, and its tools in temporary
environments. These environments are deleted after each version finishes. Scenarios use pytest,
but live outside `tests/` and are separate from the plugin's regular test suite.
The client and server use the same MLflow version; this does not test clients against different server versions.

## Selecting scenarios

```bash
# List versions without installing dependencies or accessing MongoDB.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions-all --list

# Experiments only, including the mandatory tag scenarios.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 --family experiments

# Traces and assessments across two versions.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 3.17.1 --family traces assessments

# Select a case by name, with its own server and new database.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 --case simple_attributes_artifacts_and_restart

# Preserve only failed scenarios' databases for inspection.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 --family experiments --keep-failed-db
```

Families: `experiments`, `runs`, `traces`, `assessments`, `logged_models`, `registry`,
`prompts`, `links`, `metrics`, `diagnostics`. All are selected when `--family` is omitted.
`--case` is a `pytest -k` expression, not an ID from the plan. Variants that share a structure
are parametrized; each variant receives its own server and database.
The full matrix starts many servers and can take a considerable amount of time.

`--python` selects the interpreter used for the environments. `--case-timeout` limits each
scenario (240 seconds by default), `--startup-timeout` limits server startup (90 seconds),
`--version-timeout` limits each pytest invocation (7200 seconds), and `--install-timeout`
limits each setup command (900 seconds). Process and timeout handling targets macOS/Linux.
After flushing the exporter, trace scenarios wait up to 30 seconds for the full payload and
its MongoDB spans; other public errors propagate.

## What it checks

- Experiment creation through the client and REST, lifecycle, errors, tags, and searches.
- Runs: synchronous/asynchronous logging, parameter immutability, non-finite values, metric
  history and current summaries, dataset/model inputs, outputs, status, deletion/restoration, and searches.
- Traces: spans/decorators, hierarchy, attributes, errors, metadata, late logging,
  redelivery, tags, searches, and deletion cascading to spans/assessments.
- Full retrieval from local artifacts in 3.5 or from the tracking store where applicable;
  server restarts and reads from a new client process.
- Assessments: native values, updates, metadata, errors and recovery, overrides, and deletion.
- Logged models: metadata, tags, metrics from different runs/datasets, history filters,
  metric ordering, pagination, and soft deletion.
- Registry: models/versions, descriptions, tags, renaming, stages, latest versions, aliases,
  redaction on version deletion, copies using local artifacts, and concurrent version allocation.
- Prompts: metadata/versions, tags, aliases, rendering, searches, deletion, and links.
- Aggregated trace/span/assessment metrics where upstream exposes the API: filters,
  dimensions, UTC buckets, percentiles, and available token and cost metrics.

All implemented search families have scenarios. The goal is not to cover every possible
field/comparator combination or replace the functional suite.
Results are compared with known scenario data; MongoDB is queried to verify persistence,
without using plugin pipelines as the expected result.
The `RestStore` transport used in some cases is MLflow's HTTP client; tests never instantiate
the plugin's MongoDB stores directly.

Older examples were adapted into standalone scenarios without running the original files
with their fixed URIs:

| Previous script | Equivalent scenario |
| --- | --- |
| `trace_test.py`, `test.py` | `simple_attributes_artifacts_and_restart`, `decorated_nested_script_equivalent` |
| `trace_test2.py` | `decorated_nested_script_equivalent` |
| `test3.py` | `simple_attributes_artifacts_and_restart` |
| `test4.py` | `error_script_equivalent` |
| `test5.py` | `experiment_isolation_script_equivalent` |
| `mlflow_smoke_dataset_inputs.py` | `dataset_inputs_and_fluent_script_equivalent` |

## Reports and known failures

The runner displays progress for each case. Everything is saved under `.cache/mlflow-smoke/run-*/`:

- `summary.json`: version matrix, selected filters, source fingerprint, status, duration, and counts.
- `<version>/client.log`, `install.log`, `packages.txt`, `junit.xml`, `cases.json`.
- `<version>/capabilities.json`: signatures and REST routes from the upstream installation.
- A directory per scenario containing `scenario.json`, `server.log`, `evidence.jsonl`, and artifacts.
  `scenario.json` maps the directory to its case and database;
  `server.json` records that scenario's server PID and URL.

`evidence.jsonl` records HTTP methods/routes/responses and MongoDB queries/results.
Each MongoDB query includes a `mongosh` field you can paste into the CLI: it uses the generated
database and Extended JSON to preserve types such as `ObjectId`. To inspect the database after
a failure, use `--keep-failed-db`; otherwise, it will normally have been cleaned up.
Authentication headers are not recorded. The MongoDB URI is passed to the server through the
environment; server logs replace that URI with a placeholder. Use synthetic smoke data rather than real data.
The fingerprint includes Python files from the selected plugin checkout and the smoke suite,
plus setup/pyproject files, including uncommitted changes. Changes during a version's execution
produce a `SOURCE_CHANGED` status.

Case statuses distinguish `PASSED`, `FAILED`, `SETUP_FAILED`, `UNAVAILABLE_UPSTREAM`,
and `NOT_IMPLEMENTED`. Reports preserve the phase and error for each failure. A failed scenario
does not stop other variants or families. Later steps within the same test do not execute and
are not counted as passing checks. Skipped unimplemented features are visible inventory,
not successful coverage.

The three experiment tag creation/update/deletion cases are mandatory: they have no skip or
xfail. All three passed in the first MLflow 3.5.1 run. MongoDB is also queried when the HTTP
response demonstrates a no-op. The smoke suite does not modify the store to make scenarios pass.

Persistence checks inspect document and span fields without serializing `ObjectId` values
with `json.dumps`. Numeric assessment values are compared without requiring identical
`int`/`float` types: protobuf may convert them to `double` before HTTP transport. Booleans,
strings, lists, and dictionaries retain type checks, including nested values.
Registry search fixtures create real runs and use remote sources as metadata,
without downloading or uploading S3 artifacts.

Logged-model searches specify both dataset name and digest. In 3.5.1, the HTTP handler
converts an omitted digest to `None`, which is then used as a filter; these scenarios do not
establish support for searches by name alone. Invalid logged-model limits use `-1` because
upstream normalizes `0` to the default. Invalid trace limits use `RestStore` so the request
reaches HTTP, bypassing `MlflowClient`'s early empty return for a limit of `0`.
Rejection of a rename on a deleted experiment checks the HTTP JSON directly:
`INVALID_STATE` maps to HTTP 500, and the SDK retry adapter hides that response body.

There are also two mandatory registry tag schema checks: they require `{key,value}` for
registered models and model versions, following the agreed exception for model registry
(including prompts). Tracking store entries continue to use `{k,v}`.
The runner does not change schemas or migrate data. Functional tag cases check content
separately; they do not replace the schema checks.

Prompt searches use filters and pagination without assuming `search_prompts` accepts
`order_by`; names are compared independently of their order. A missing prompt alias must
return `INVALID_PARAMETER_VALUE`. Model-version timestamp filters run only when the upstream
validator supports both fields; in 3.5.1, they are marked `UNAVAILABLE_UPSTREAM` before
creating the scenario's server and database.

APIs missing from an upstream version are skipped with a specific reason. Advanced trace
filters and `model_id`, logged-model finalization, and trace unlinking are recorded as plugin
limitations. Unexpected exceptions are not converted into skips.
`diagnostics` checks the empty `search_datasets` contract; this does not establish support
for evaluation dataset storage.

The exit code is 0 when all selected, executed cases pass; 1 for failures or environment
problems; and 130 when interrupted. Also review the counts of unavailable APIs and
unimplemented features: a zero exit code does not mean full MLflow parity.

Normal cleanup and Ctrl+C stop the server and delete only the database generated for that
scenario. `--keep-failed-db` preserves the failed database and records its name while keeping
the server stopped. A forced system shutdown may prevent teardown: the exact database and
directory names remain in the reports for inspection.

## Limitations

The first real MLflow 3.5.1 run produced 133 passing cases, 38 failures, and 20 setup failures,
plus 44 unavailable APIs and 4 unimplemented features. Smoke fixtures and assertions identified
in that run were corrected and rerun: the second run produced 167 passing cases, 24 failures,
and no setup failures, with the same 44 unavailable upstream cases and 4 unimplemented cases.
After correcting the smoke suite and the plugin's placeholder conflicts, protobuf list handling,
and assessment updates, a full 3.5.1 run passed 190 cases with no failures:
45 were marked `UNAVAILABLE_UPSTREAM` and 4 `NOT_IMPLEMENTED`. This result does not establish
support for other versions or skipped features. Reading/parsing source code does not verify
the HTTP contract for each MLflow version.
The [plan](PLAN.md) maintains the broader inventory, and [coverage](COVERAGE.md) distinguishes
implemented scenarios from extensions.
The suite does not implement plugin fixes, a custom artifact provider, S3/proxy support,
workspaces, webhooks, trace archiving, driver failures, BSON limits, or load testing.
