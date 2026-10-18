"""Run the HTTP/Mongo smoke matrix without changing the caller's environment."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def release(value):
    if not re.fullmatch(r"3\.(?:[5-9]|[1-9]\d+)\.(?:0|[1-9]\d*)", value):
        raise argparse.ArgumentTypeError("Expected a stable MLflow 3.x version >=3.5.0")
    return value


def discover():
    request = Request(
        "https://pypi.org/simple/mlflow/",
        headers={
            "Accept": "application/vnd.pypi.simple.v1+json",
            "User-Agent": "mlflow-mongodb-smoke",
        },
    )
    with urlopen(request, timeout=30) as response:
        distributions = json.load(response)["files"]
    versions = set()
    for item in distributions:
        match = re.match(r"mlflow-(3\.(?:[5-9]|[1-9]\d+)\.\d+)(?:-|\.tar\.gz$)", item["filename"])
        if match and not item.get("yanked", False):
            versions.add(match[1])
    if not versions:
        raise ValueError("No stable releases >=3.5 were found")
    return sorted(versions, key=lambda value: tuple(map(int, value.split("."))))


def fingerprint(plugin_root):
    digest = hashlib.sha256()
    sources = {
        "plugin": (plugin_root, [plugin_root / "setup.py", plugin_root / "pyproject.toml"]
                   + sorted((plugin_root / "mlflow_mongodb").rglob("*.py"))),
        "smoke": (ROOT, sorted((ROOT / "smoke_test").rglob("*.py"))),
    }
    for name, (root, paths) in sources.items():
        for path in paths:
            if path.is_file():
                digest.update(f"{name}/{path.relative_to(root)}".encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def command(argv, log, env, timeout, live=False):
    # Do not log environment variables: Mongo credentials are passed only there.
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE if live else log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    reader = None
    if live:

        def drain():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        if reader is not None:
            reader.join(timeout=5)
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plugin-dir", type=Path,
        help="Local mlflow-mongodb checkout to install and inspect (required unless --list)",
    )
    parser.add_argument("--versions", nargs="+", type=release)
    parser.add_argument(
        "--versions-all", action="store_true", help="All stable patches >=3.5 (default)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List versions without installing/running"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Use Python 3.10+; 3.11 recommended"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/mlflow-smoke")
    parser.add_argument(
        "--family",
        nargs="+",
        choices=[
            "experiments",
            "runs",
            "traces",
            "assessments",
            "logged_models",
            "registry",
            "prompts",
            "links",
            "metrics",
            "diagnostics",
        ],
    )
    parser.add_argument("--keep-failed-db", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=90)
    parser.add_argument("--case-timeout", type=int, default=240)
    parser.add_argument("--version-timeout", type=int, default=7200)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--case", help="Optional pytest -k expression")
    args = parser.parse_args()
    if args.versions and args.versions_all:
        parser.error("Choose --versions or --versions-all")
    try:
        versions = list(dict.fromkeys(args.versions or discover()))
    except (OSError, ValueError, KeyError):
        parser.error(
            "Cannot discover MLflow versions from PyPI. Check connectivity or use --versions."
        )
    if min(args.case_timeout, args.startup_timeout, args.version_timeout, args.install_timeout) < 1:
        parser.error("Timeouts must be positive integers")
    if args.list:
        print("\n".join(versions))
        return 0
    if args.plugin_dir is None:
        parser.error("Specify --plugin-dir with the local mlflow-mongodb checkout")
    plugin_root = args.plugin_dir.expanduser().resolve()
    if not (plugin_root / "mlflow_mongodb").is_dir() or not any(
        (plugin_root / name).is_file() for name in ("setup.py", "pyproject.toml")
    ):
        parser.error("--plugin-dir must contain mlflow_mongodb and setup.py or pyproject.toml")
    if not os.environ.get("MONGODB_URI"):
        parser.error("Set MONGODB_URI. Its database is ignored; only generated databases are used.")
    interpreter = shutil.which(args.python)
    if not interpreter:
        parser.error("Python interpreter not found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir)).resolve()
    report = {
        "versions": versions,
        "python": interpreter,
        "source_sha256": fingerprint(plugin_root),
        "plugin_directory": str(plugin_root),
        "families": args.family,
        "case_filter": args.case,
        "results": [],
    }
    print(f"Versions: {', '.join(versions)}\nReports: {output}", flush=True)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    interrupted = False
    for version in versions:
        target = output / version
        target.mkdir()
        started = time.monotonic()
        result = {
            "mlflow": version,
            "status": "SETUP_FAILED",
            "directory": str(target),
            "source_sha256": fingerprint(plugin_root),
        }
        try:
            with tempfile.TemporaryDirectory(prefix=f"mlflow-smoke-{version}-") as temporary:
                venv = Path(temporary) / "venv"
                executable = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
                with (target / "install.log").open("w") as log:
                    if command(
                        [interpreter, "-m", "venv", str(venv)], log, env, args.install_timeout
                    ):
                        raise RuntimeError("venv creation failed; see install.log")
                    install = [
                        executable,
                        "-m",
                        "pip",
                        "install",
                        "-e",
                        str(plugin_root),
                        f"mlflow=={version}",
                        "pytest==8.4.2",
                        "pytest-timeout==2.4.0",
                        "requests",
                        "numpy",
                    ]
                    if command(install, log, env, args.install_timeout):
                        raise RuntimeError("dependency installation failed; see install.log")
                with (target / "packages.txt").open("w") as log:
                    if command([executable, "-m", "pip", "freeze"], log, env, 60):
                        raise RuntimeError("pip freeze failed")
                worker_env = {
                    key: value
                    for key, value in env.items()
                    if not key.startswith(("MLFLOW_", "SMOKE_"))
                }
                worker_env.update(
                    {
                        "SMOKE_OUTPUT": str(target),
                        "SMOKE_STARTUP_TIMEOUT": str(args.startup_timeout),
                        "SMOKE_KEEP_FAILED_DB": str(int(args.keep_failed_db)),
                        "SMOKE_FAMILIES": ",".join(args.family or []),
                        "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "true",
                        "MLFLOW_HTTP_REQUEST_TIMEOUT": "15",
                        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
                        "MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT": "15",
                        "MLFLOW_ENABLE_TELEMETRY": "false",
                        "DO_NOT_TRACK": "1",
                        "PATH": str(Path(executable).parent) + os.pathsep + env.get("PATH", ""),
                        "VIRTUAL_ENV": str(venv),
                        "PYTEST_ADDOPTS": "",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                        "PYTEST_PLUGINS": "",
                        "PYTHONOPTIMIZE": "0",
                    }
                )
                argv = [
                    executable,
                    "-m",
                    "pytest",
                    str(ROOT / "smoke_test/cases"),
                    "-p",
                    "pytest_timeout",
                    "-v",
                    "--tb=short",
                    "--timeout-method=signal",
                    f"--timeout={args.case_timeout}",
                    "-o",
                    "addopts=",
                    f"--junitxml={target / 'junit.xml'}",
                ]
                if args.case:
                    argv += ["-k", args.case]
                with (target / "client.log").open("w") as log:
                    code = command(argv, log, worker_env, args.version_timeout, live=True)
                result.update(exit_code=code, status="PASSED" if code == 0 else "FAILED")
                cases_report = target / "cases.json"
                if cases_report.exists():
                    result["cases"] = json.loads(cases_report.read_text())
                    result["counts"] = dict(Counter(item["status"] for item in result["cases"]))
                if not result.get("cases"):
                    result["status"] = "SETUP_FAILED" if code != 5 else "NO_CASES_SELECTED"
                if result["source_sha256"] != fingerprint(plugin_root):
                    result["status"] = "SOURCE_CHANGED"
        except KeyboardInterrupt:
            interrupted = True
            result["status"] = "INTERRUPTED"
        except subprocess.TimeoutExpired:
            result["status"] = "TIMEOUT"
        except RuntimeError as exc:
            result["error"] = str(exc)
        except OSError as exc:
            result["error"] = f"Process setup failed ({type(exc).__name__})"
        # Preserve completed case evidence even after a timeout or Ctrl+C.
        cases_report = target / "cases.json"
        if cases_report.exists() and "cases" not in result:
            result["cases"] = json.loads(cases_report.read_text())
            result["counts"] = dict(Counter(item["status"] for item in result["cases"]))
        result["seconds"] = round(time.monotonic() - started, 2)
        report["results"].append(result)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{version}: {result['status']} ({result['seconds']}s)", flush=True)
        if interrupted:
            break
    print(f"Summary: {output / 'summary.json'}")
    return (
        130 if interrupted else int(any(item["status"] != "PASSED" for item in report["results"]))
    )


if __name__ == "__main__":
    raise SystemExit(main())
