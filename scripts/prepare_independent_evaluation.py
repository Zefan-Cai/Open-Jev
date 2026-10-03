"""Create a pinned 27B evaluator handoff without loading models or running an eval.

The generated scripts serve the released checkpoint and optionally reproduce
the historical PUBLIC 231-task protocol. Sealed evaluation belongs to the
independent evaluator; preparing this bundle never claims a sealed result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "ZefanCai/Open-Jev-27B-v1.1"
MODEL_REVISION = "28cf73067d5b337860bbef3c85b8b82ba8730956"
BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CHECKPOINT_SHA256 = "c49994563c3c4f04a99d9130203c4e526f4ae5086c84deec57698d18cb652e71"
TEMPERATURE = 2.5343690298472983
BENCHMARK_REVISION = "f8ce71361165846101d02ebc83ad44e47ae44fc3"
DEPENDENCY_LOCK = ROOT / "requirements-linux-py311-cu128-20261002.lock"


# Runs on the evaluator's host after download, before any model allocation.
VERIFIER = '''"""Verify the isolated runtime and checkpoint bytes, without importing a model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import site
import subprocess
import sys


def verify_lock(root, manifest):
    contract = manifest["runtime_contract"]
    for name in ("requirements.lock", "requirements-pypi.lock"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != contract["files_sha256"][name]:
            raise SystemExit("Dependency lock checksum differs")


def verify_runtime(root, manifest):
    verify_lock(root, manifest)
    contract = manifest["runtime_contract"]
    if (sys.version_info[:2] != (3, 11) or platform.system() != "Linux"
            or platform.machine() != "x86_64"):
        raise SystemExit("Dependency lock requires Linux x86-64 and Python 3.11")
    if sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
        raise SystemExit("Evaluator runtime must use an isolated venv")
    prefix = Path(sys.prefix).resolve()
    packages = {}
    for line in (root / "requirements.lock").read_text().splitlines():
        name, expected = line.split("==")
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit("Missing locked package: " + name)
        if distribution.version != expected:
            raise SystemExit("Locked package version differs: " + name)
        if not Path(distribution.locate_file("")).resolve().is_relative_to(prefix):
            raise SystemExit("Locked package is outside the venv: " + name)
        packages[name] = distribution.version
    subprocess.run([sys.executable, "-I", "-m", "pip", "--isolated", "check"],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"runtime_contract_verified": True, "dependency_lock_sha256":
            contract["files_sha256"]["requirements.lock"], "pip_check": "passed",
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": packages, "inference_performed": False,
            "hardware_test_status": "not_measured",
            "sealed_result_status": "pending_independent_evaluator"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runtime-only", action="store_true")
    group.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "submission.json").read_text())
    if args.lock_only:
        verify_lock(root, manifest)
        print(json.dumps({"dependency_lock_verified": True, "inference_performed": False}))
        return
    report = verify_runtime(root, manifest)
    if args.runtime_only:
        (root / "runtime-preflight.json").write_text(json.dumps(report, indent=2) + "\\n")
        print(json.dumps(report, indent=2))
        return
    checkpoint = root / "model/package/checkpoint"
    digest = hashlib.sha256()
    for file in sorted(checkpoint.rglob("*")):
        if file.is_symlink():
            raise SystemExit("Checkpoint symlinks are not accepted")
        if file.is_file():
            digest.update(file.relative_to(checkpoint).as_posix().encode() + b"\\0")
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    if digest.hexdigest() != manifest["identity"]["checkpoint_sha256"]:
        raise SystemExit("Checkpoint tree checksum differs from the published release")
    model = json.loads((checkpoint / "model.json").read_text())
    temperature = json.loads((checkpoint / "temperature.json").read_text())["temperature"]
    if (model["model_id"] != manifest["identity"]["model"]
            or model["revision"] != manifest["identity"]["base_revision"]
            or temperature != manifest["identity"]["temperature"]):
        raise SystemExit("Checkpoint model, base revision or temperature differs")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(root / "loader"), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root / "loader"), "status", "--porcelain"], text=True)
    if actual_commit != manifest["identity"]["code_commit"] or dirty:
        raise SystemExit("Evaluator loader must be the pinned clean checkout")
    try:
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    report.update(checkpoint_verified=True, loader_commit=actual_commit, gpu_inventory=gpu)
    (root / "runtime.json").write_text(json.dumps(report, indent=2) + "\\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
'''


def prepare(output, code_revision, *, max_length=16384):
    if re.fullmatch(r"[0-9a-f]{40}", code_revision) is None:
        raise ValueError("Use a full 40-character loader commit")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    lock = DEPENDENCY_LOCK.read_text()
    pypi_lock = "".join(line + "\n" for line in lock.splitlines()
                        if not line.startswith("torch=="))
    runtime_files = {"requirements.lock": lock, "requirements-pypi.lock": pypi_lock}
    identity = {"model": BASE_MODEL, "method": "lora_decision_head",
                "base_revision": BASE_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                "temperature": TEMPERATURE, "code_commit": code_revision,
                "max_length": max_length}
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_evaluated", "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION, "identity": identity,
        "request_settings": {"endpoint": "http://127.0.0.1:8791/v1/systemone",
                             "device": "cuda:0", "batch_size": 1, "prefix_cache": False,
                             "concurrency": 1, "probabilities": "native",
                             "saved_training_max_length": 4096,
                             "evaluation_max_length_override": max_length != 4096},
        "hardware": {"status": "not_measured", "gpu_inventory": [],
                     "note": "Evaluator must record hardware and sufficient BF16 27B memory. No single-GPU minimum or speedup is established by this bundle."},
        "public_reproduction": {"benchmark_revision": BENCHMARK_REVISION,
                                "denominators": {"original": 72, "easy": 48, "hard": 111,
                                                 "total": 231},
                                "warmups": 0, "retries": 0, "status": "not_run"},
        "independent_evaluation": {
            "reference_snapshot": "JevBench v1.5.4",
            "reference_denominators": {"open": 904, "sealed": 720, "total": 1624},
            "status": "pending_independent_evaluator", "results": None,
            "note": "Reference counts are not attempted requests. Evaluator owns its exact protocol and sealed tasks. Request aggregate results and a run manifest, never private questions or labels."},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_contract": {
            "source_lock": DEPENDENCY_LOCK.name,
            "platform": "linux-x86_64", "python": "3.11", "torch_cuda": "12.8",
            "files_sha256": {name: hashlib.sha256(value.encode()).hexdigest()
                             for name, value in runtime_files.items()},
            "note": "Observed version lock, not an artifact-hash lock or a completed install replay. Preparation performs no installation or model calls.",
        },
    }
    start = f'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME
export PIP_CONFIG_FILE=/dev/null
test ! -e .venv && test ! -e loader && test ! -e model
python3.11 -I -c 'import platform, sys; assert sys.version_info[:2] == (3, 11) and platform.system() == "Linux" and platform.machine() == "x86_64", "Requires Linux x86-64 / Python 3.11"'
python3.11 -I verify_and_record.py --lock-only
git clone https://github.com/Zefan-Cai/Open-Jev.git loader
git -C loader checkout --detach {code_revision}
python3.11 -I -m venv .venv
.venv/bin/python -I -m pip --isolated install --no-deps \\
  --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128'
.venv/bin/python -I -m pip --isolated install --no-deps \\
  --index-url https://pypi.org/simple --requirement requirements-pypi.lock
.venv/bin/python -I -m pip --isolated install --no-deps --no-build-isolation -e './loader[train]'
.venv/bin/python -I verify_and_record.py --runtime-only
.venv/bin/hf download {MODEL_REPO} --revision {MODEL_REVISION} --local-dir model
.venv/bin/python -I verify_and_record.py
cd loader
exec ../.venv/bin/python -I -m jev.server --checkpoint ../model/package/checkpoint \\
  --device cuda:0 --max-length {max_length} --batch-size 1 --no-prefix-cache \\
  --host 127.0.0.1 --port 8791
'''
    public = f'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d jevbench ]; then
  git clone https://github.com/fstandhartinger/jevbench.git jevbench
  git -C jevbench checkout --detach {BENCHMARK_REVISION}
fi
if [ {max_length} -ne 16384 ]; then
  echo 'Public reproduction requires --max-length 16384; prepare a separate bundle.' >&2
  exit 1
fi
EVAL_ROOT="$PWD"
cd loader
"$EVAL_ROOT/.venv/bin/python" -m scripts.jevbench_openjev prepare \\
  --upstream "$EVAL_ROOT/jevbench" --output "$EVAL_ROOT/public-inputs"
REQUEST_SHA=$("$EVAL_ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["requests_sha256"])' "$EVAL_ROOT/public-inputs/manifest.json")
"$EVAL_ROOT/.venv/bin/python" -m scripts.jevbench_openjev collect \\
  --requests "$EVAL_ROOT/public-inputs/requests.json" --input-sha256 "$REQUEST_SHA" \\
  --endpoint http://127.0.0.1:8791/v1/systemone --identity "$EVAL_ROOT/identity.json" \\
  --output "$EVAL_ROOT/public-run" --timeout 300 --max-seconds 14400
"$EVAL_ROOT/.venv/bin/python" -m scripts.jevbench_openjev summarize \\
  --upstream "$EVAL_ROOT/jevbench" --run-dir "$EVAL_ROOT/public-run" \\
  --output "$EVAL_ROOT/public-aggregate.json"
'''
    readme = f'''# Open-Jev-27B-v1.1 independent evaluator handoff

Prepared input only. No new public or sealed result has been produced.

On a Linux NVIDIA host with sufficient BF16 27B memory, run `bash install-and-serve.sh`.
This pins loader `{code_revision}`, model package `{MODEL_REVISION}` and base
`{BASE_MODEL}@{BASE_REVISION}`. Base weights download on the first model load.
The checkpoint checksum and a clean loader checkout are verified before inference.
The installer requires Linux x86-64 and Python 3.11 and checks both carried
lock hashes before installation. It disables pip configuration files with
`PIP_CONFIG_FILE=/dev/null`, then installs Torch from the CUDA index, all
remaining locked distributions from PyPI, and the pinned loader without
dependency resolution or build isolation. The metadata-only runtime preflight
checks both lock hashes, all locked package versions and venv locations, and
`pip check` before any model allocation. `runtime-preflight.json` records that
check; `runtime.json` adds checkpoint identity and GPU inventory before serving.
The version lock does not hash downloaded package artifacts, and preparing
this bundle is not a completed clean-install replay or hardware performance
result. No credentials are recorded.

The service is loopback-only. In another terminal, check readiness and submit
the released example request:

```bash
curl --fail http://127.0.0.1:8791/health
curl --fail http://127.0.0.1:8791/v1/systemone \\
  -H 'Content-Type: application/json' --data-binary @loader/configs/example-request.json
```

For the historical 231-public protocol only, run `bash reproduce-public.sh`
after the server is ready. The original 72, easy 48 and hard 111 counts remain
separate from newer JevBench versions. No warmups or retries, concurrency one,
uncached native probabilities. `{max_length}` is the requested token limit;
the saved training limit is 4,096. A larger evaluation limit is an explicit
override and does not establish equivalent quality at longer contexts.
Public input/raw-response files remain local: upstream source licensing is
not uniform. Publish only the aggregate and this submission/runtime manifest.

For a fresh independent sealed evaluation, the evaluator runs its own frozen
suite against `/v1/systemone`. The v1.5.4 reference had 904 open + 720 sealed
tasks; these are planning context, not results. Record actual denominators,
attempted/failed/missing requests, primitive metrics, calibration, abstentions,
request settings, checkpoint identity, measured hardware/time and estimated
cost separately. A raw probability interval and abstention policy must be
reported alongside raw accuracy. No private questions or labels are requested.
All sealed results remain pending until the independent evaluator supplies
an aggregate and reproducible manifest.
'''
    files = {"submission.json": json.dumps(manifest, indent=2) + "\n",
             "identity.json": json.dumps(identity, indent=2) + "\n",
             "README.md": readme, "install-and-serve.sh": start,
             "reproduce-public.sh": public, "verify_and_record.py": VERIFIER,
             **runtime_files}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        (output / name).write_text(value)
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files}
    (output / "files.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", help="Full loader commit; default is this checkout's HEAD")
    parser.add_argument("--max-length", type=int, default=16384,
                        help="Explicit evaluation context limit; saved training limit is 4096")
    args = parser.parse_args()
    revision = args.code_revision or subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = prepare(args.output, revision, max_length=args.max_length)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "sealed_status": result["independent_evaluation"]["status"]}))


if __name__ == "__main__":
    main()
