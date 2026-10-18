# MLflow MongoDB smoke test

HTTP and MongoDB smoke tests for a local [mlflow-mongodb](https://github.com/WaVEV/mlflow-mongodb) checkout across MLflow 3.5+ versions.

The runner creates a temporary environment for each version, installs the selected plugin checkout, and starts isolated servers and MongoDB databases for each scenario. It requires Python 3.11 or 3.12, MongoDB 8.0+, and access to PyPI.

```bash
export MONGODB_URI='mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000'
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1

# All stable versions starting with 3.5
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions-all
```

`--plugin-dir` points to the checkout you want to test; the runner does not change its branch or source files. Dependencies are installed in temporary environments. Results are saved under `.cache/mlflow-smoke/`, which is excluded from Git.

See the [full instructions](smoke_test/README.md), [coverage inventory](smoke_test/COVERAGE.md), and [plan](smoke_test/PLAN.md).
