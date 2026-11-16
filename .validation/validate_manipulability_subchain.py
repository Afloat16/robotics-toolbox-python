import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

kind = sys.argv[1]
base = "cee0023535988e4ae9490c81d639a46f2311b8fd"
candidate = "c4e6e28d0fd748b46b4059faf52ed19f2b0e0048"
output = Path(".validation/manipulability-evidence")
output.mkdir(parents=True, exist_ok=True)
records = []

def run(name, args, timeout=1200):
    print("VALIDATION_COMMAND " + json.dumps({"name": name, "args": args}), flush=True)
    started = time.monotonic()
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        text, code = result.stdout, result.returncode
    except subprocess.TimeoutExpired as error:
        raw = error.stdout or b""
        text = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        text += "\nExternal validation timeout expired.\n"
        code = 124
    (output / (name + ".log")).write_text(text)
    print(text, flush=True)
    record = {"name": name, "args": args, "exit_code": code,
              "elapsed_seconds": time.monotonic() - started}
    junit = output / (name + ".xml")
    if junit.exists():
        tree = ET.parse(junit)
        suites = list(tree.getroot().iter("testsuite"))
        record["junit"] = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "errors", "failures", "skipped")
        }
        record["unsuccessful_cases"] = [
            case.attrib.get("classname", "") + "::" + case.attrib["name"]
            for case in tree.getroot().iter("testcase")
            if case.find("failure") is not None or case.find("error") is not None
        ]
    records.append(record)
    return code

pytest = [sys.executable, "-m", "pytest"]
regression_name = kind + "-regressions"
regression = run(regression_name, pytest + [
    "tests/test_manipulability_subchain.py", "-q", "--tb=short",
    "--timeout=50", "--timeout_method=thread",
    "--junitxml=" + str(output / (regression_name + ".xml"))
])

core = ["tests/", "--ignore=tests/test_blocks.py", "-m", "not slow",
        "--timeout=50", "--timeout_method=thread", "-q", "--tb=short"]
if kind == "baseline":
    core += ["--ignore=tests/test_manipulability_subchain.py"]
canonical_name = kind + "-canonical-core"
canonical = run(canonical_name, pytest + core + [
    "--junitxml=" + str(output / (canonical_name + ".xml"))
])
filtered = None
if canonical != 0:
    filtered_name = kind + "-core-excluding-known-fdyn-timeout"
    filtered = run(filtered_name, pytest + core + [
        "--deselect=tests/test_dynamics_defaults.py::test_fdyn_without_optional_arguments",
        "--junitxml=" + str(output / (filtered_name + ".xml"))
    ])
mypy = run(kind + "-mypy", [sys.executable, "-m", "mypy"])
lint = run(kind + "-regression-lint", [
    sys.executable, "-m", "ruff", "check", "tests/test_manipulability_subchain.py"
])
formatting = run(kind + "-regression-format", [
    sys.executable, "-m", "ruff", "format", "--check", "tests/test_manipulability_subchain.py"
])
summary = {
    "kind": kind, "base": base, "candidate": candidate,
    "records": records,
    "canonical_core_passed": canonical == 0,
    "filtered_core_passed": filtered == 0 if filtered is not None else None,
    "type_check_exit_code": mypy,
    "note": "The filtered run excludes only the named previously observed fdyn timeout; canonical outcomes are preserved separately."
}
(output / (kind + "-summary.json")).write_text(json.dumps(summary, indent=2))
print("MANIPULABILITY_VALIDATION_SUMMARY " + json.dumps(summary), flush=True)
expected_regression = 1 if kind == "baseline" else 0
raise SystemExit(0 if regression == expected_regression and
                 (canonical == 0 or filtered == 0) and lint == 0 and formatting == 0 else 1)
