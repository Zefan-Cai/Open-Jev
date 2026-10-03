"""Submission provenance checks; synthetic file fixtures are not model weights."""
from contextlib import ExitStack, redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_independent_evaluation as submission
from scripts.evaluate_openjev_provider import validate_identity_config


class IndependentEvaluationTests(unittest.TestCase):
    def run_verifier(self, output, *, runtime_only=False, distributions=None,
                     python_version=(3, 11, 15), system="Linux", machine="x86_64",
                     pip_error=None, venv=True, user_site=False):
        prefix = output / ".venv"
        versions = dict(line.split("==") for line in
                        (output / "requirements.lock").read_text().splitlines())
        def distribution(name):
            if distributions:
                return distributions(name, versions[name], prefix)
            return SimpleNamespace(version=versions[name], locate_file=lambda path: prefix / "lib")
        def command(args, **kwargs):
            return "a" * 40 + "\n" if "rev-parse" in args else ""
        with ExitStack() as mocks:
            mocks.enter_context(patch("sys.argv", [str(output / "verify_and_record.py")] +
                                      (["--runtime-only"] if runtime_only else [])))
            mocks.enter_context(patch("sys.version_info", python_version))
            mocks.enter_context(patch("sys.prefix", str(prefix)))
            mocks.enter_context(patch("sys.base_prefix", "/fixture-base" if venv else str(prefix)))
            mocks.enter_context(patch("site.ENABLE_USER_SITE", user_site))
            mocks.enter_context(patch("platform.system", return_value=system))
            mocks.enter_context(patch("platform.machine", return_value=machine))
            mocks.enter_context(patch("platform.platform", return_value="Linux-fixture"))
            mocks.enter_context(patch("importlib.metadata.distribution", side_effect=distribution))
            mocks.enter_context(patch("subprocess.check_output", side_effect=command))
            pip_check = mocks.enter_context(patch("subprocess.run", side_effect=pip_error))
            mocks.enter_context(redirect_stdout(StringIO()))
            runpy.run_path(str(output / "verify_and_record.py"), run_name="__main__")
            return pip_check

    def test_invalid_revision_cannot_write_or_inject_shell_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            for revision in ("main", "a" * 39, "$(touch injected)", "a" * 40 + "\n"):
                with self.assertRaises(ValueError):
                    submission.prepare(output, revision)
                self.assertFalse(output.exists())
            for length in (0, -1, True):
                with self.assertRaises(ValueError):
                    submission.prepare(output, "a" * 40, max_length=length)
                self.assertFalse(output.exists())

    def test_bundle_is_pinned_not_evaluated_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            manifest = submission.prepare(output, "a" * 40)
            validate_identity_config(manifest["identity"])
            self.assertIsNone(manifest["independent_evaluation"]["results"])
            self.assertEqual(manifest["status"], "prepared_not_evaluated")
            self.assertEqual(manifest["public_reproduction"]["denominators"]["total"], 231)
            self.assertEqual(manifest["request_settings"]["saved_training_max_length"], 4096)
            self.assertTrue(manifest["request_settings"]["evaluation_max_length_override"])
            self.assertEqual((output / "requirements.lock").read_bytes(),
                             submission.DEPENDENCY_LOCK.read_bytes())
            self.assertEqual(manifest["runtime_contract"]["python"], "3.11")
            self.assertEqual(manifest["runtime_contract"]["platform"], "linux-x86_64")
            self.assertNotIn("torch==", (output / "requirements-pypi.lock").read_text())
            install = (output / "install-and-serve.sh").read_text()
            self.assertIn("python3.11 -I -m venv", install)
            self.assertIn("--no-deps --no-build-isolation", install)
            self.assertIn("torch==2.8.0+cu128", install)
            self.assertLess(install.index("export PIP_CONFIG_FILE=/dev/null"),
                            install.index("-m pip"))
            self.assertLess(install.index("--lock-only"), install.index("-m pip"))
            self.assertLess(install.index("--runtime-only"), install.index("hf download"))
            self.assertLess(install.index("--runtime-only"), install.index("-m jev.server"))
            hashes = json.loads((output / "files.json").read_text())
            for name, checksum in hashes.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), checksum)
            for script in ("install-and-serve.sh", "reproduce-public.sh"):
                subprocess.run(["bash", "-n", str(output / script)], check=True)
            before = (output / "submission.json").read_bytes()
            with self.assertRaises(FileExistsError):
                submission.prepare(output, "b" * 40)
            self.assertEqual((output / "submission.json").read_bytes(), before)

    def test_verifier_rejects_wrong_bytes_without_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            submission.prepare(output, "a" * 40)
            checkpoint = output / "model/package/checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "head.pt").write_bytes(b"corrupted opaque fixture")
            with self.assertRaisesRegex(SystemExit, "checksum differs"):
                self.run_verifier(output)
            self.assertFalse((output / "runtime.json").exists())

    def test_verified_fixture_still_does_not_claim_inference_or_sealed_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            manifest = submission.prepare(output, "a" * 40, max_length=4096)
            checkpoint = output / "model/package/checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.json").write_text(json.dumps({
                "model_id": submission.BASE_MODEL, "revision": submission.BASE_REVISION}))
            (checkpoint / "temperature.json").write_text(json.dumps({
                "temperature": submission.TEMPERATURE}))
            digest = hashlib.sha256()
            for file in sorted(checkpoint.rglob("*")):
                if file.is_file():
                    digest.update(file.relative_to(checkpoint).as_posix().encode() + b"\0")
                    digest.update(file.read_bytes())
            # This positive fixture uses its own checksum, never the release checksum.
            manifest["identity"]["checkpoint_sha256"] = digest.hexdigest()
            (output / "submission.json").write_text(json.dumps(manifest))
            self.run_verifier(output)
            report = json.loads((output / "runtime.json").read_text())
            self.assertTrue(report["checkpoint_verified"])
            self.assertFalse(report["inference_performed"])
            self.assertEqual(report["hardware_test_status"], "not_measured")
            self.assertEqual(report["sealed_result_status"], "pending_independent_evaluator")
            self.assertTrue(report["runtime_contract_verified"])
            self.assertEqual(report["packages"]["torch"], "2.8.0+cu128")

    def test_runtime_only_preflight_needs_no_checkpoint_or_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            submission.prepare(output, "a" * 40)
            before = {name for name in sys.modules if name.split(".")[0] in
                      ("torch", "transformers", "peft", "accelerate")}
            pip_check = self.run_verifier(output, runtime_only=True)
            after = {name for name in sys.modules if name.split(".")[0] in
                     ("torch", "transformers", "peft", "accelerate")}
            self.assertEqual(before, after)
            self.assertFalse((output / "model").exists())
            self.assertFalse((output / "runtime.json").exists())
            report = json.loads((output / "runtime-preflight.json").read_text())
            self.assertEqual(len(report["packages"]),
                             len(submission.DEPENDENCY_LOCK.read_text().splitlines()))
            self.assertTrue(report["runtime_contract_verified"])
            self.assertFalse(report["inference_performed"])
            self.assertNotIn("checkpoint_verified", report)
            pip_check.assert_called_once_with(
                [sys.executable, "-I", "-m", "pip", "--isolated", "check"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_both_dependency_lock_files_reject_tamper_before_checkpoint(self):
        for name in ("requirements.lock", "requirements-pypi.lock"):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "bundle"
                submission.prepare(output, "a" * 40)
                with (output / name).open("a") as stream:
                    stream.write("injected==1.0\n")
                with self.assertRaisesRegex(SystemExit, "Dependency lock checksum differs"):
                    self.run_verifier(output, runtime_only=True)
                self.assertFalse((output / "runtime-preflight.json").exists())

    def test_runtime_rejects_package_version_drift_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            submission.prepare(output, "a" * 40)
            def drift(name, version, prefix):
                return SimpleNamespace(version="99.0" if name == "torch" else version,
                                       locate_file=lambda path: prefix / "lib")
            with self.assertRaisesRegex(SystemExit, "Locked package version differs: torch"):
                self.run_verifier(output, distributions=drift)
            self.assertFalse((output / "runtime.json").exists())

    def test_runtime_rejects_missing_or_external_packages(self):
        import importlib.metadata
        for failure in ("missing", "external"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "bundle"
                submission.prepare(output, "a" * 40)
                def invalid(name, version, prefix):
                    if failure == "missing":
                        raise importlib.metadata.PackageNotFoundError(name)
                    return SimpleNamespace(version=version, locate_file=lambda path: Path("/system/lib"))
                with self.assertRaisesRegex(SystemExit, "Missing locked package|outside the venv"):
                    self.run_verifier(output, runtime_only=True, distributions=invalid)
                self.assertFalse((output / "runtime-preflight.json").exists())

    def test_runtime_rejects_unsupported_interpreter_or_platform(self):
        for overrides in ({"python_version": (3, 12, 0)}, {"system": "Darwin"},
                          {"machine": "aarch64"}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "bundle"
                submission.prepare(output, "a" * 40)
                with self.assertRaisesRegex(SystemExit, "requires Linux x86-64 and Python 3.11"):
                    self.run_verifier(output, runtime_only=True, **overrides)
                self.assertFalse((output / "runtime-preflight.json").exists())

    def test_runtime_rejects_broken_dependency_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            submission.prepare(output, "a" * 40)
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_verifier(output, runtime_only=True,
                                  pip_error=subprocess.CalledProcessError(1, "pip check"))
            self.assertFalse((output / "runtime-preflight.json").exists())

    def test_runtime_rejects_global_or_user_site_environment(self):
        for overrides in ({"venv": False}, {"user_site": True}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "bundle"
                submission.prepare(output, "a" * 40)
                with self.assertRaisesRegex(SystemExit, "must use an isolated venv"):
                    self.run_verifier(output, runtime_only=True, **overrides)
                self.assertFalse((output / "runtime-preflight.json").exists())

    def test_install_script_disables_inherited_pip_config_before_first_pip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            submission.prepare(output, "a" * 40)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            stub = root / "venv-python"
            capture = root / "first-pip.txt"
            dangerous_config = root / "pip.conf"
            dangerous_config.write_text("[global]\nindex-url=https://redirect.invalid/simple\n")
            # Host/platform, clone and venv commands are inert fixtures. The first
            # pip stub captures its environment and exits; no install or network.
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$PIP_CONFIG_FILE" "$@" > "$JEV_PIP_CAPTURE"\nexit 73\n')
            (fake_bin / "python3.11").write_text(
                '#!/usr/bin/env bash\nif [ "${3:-}" = venv ]; then\n'
                '  mkdir -p "$4/bin"\n  cp "$JEV_PIP_STUB" "$4/bin/python"\nfi\n')
            (fake_bin / "git").write_text(
                '#!/usr/bin/env bash\nif [ "$1" = clone ]; then mkdir "$3"; fi\n')
            for executable in (stub, fake_bin / "python3.11", fake_bin / "git"):
                executable.chmod(0o755)
            env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                   "PIP_CONFIG_FILE": str(dangerous_config), "JEV_PIP_CAPTURE": str(capture),
                   "JEV_PIP_STUB": str(stub)}
            completed = subprocess.run(["bash", str(output / "install-and-serve.sh")],
                                       env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 73, completed.stderr)
            first_pip = capture.read_text().splitlines()
            self.assertEqual(first_pip[0], "/dev/null")
            self.assertEqual(first_pip[1:6], ["-I", "-m", "pip", "--isolated", "install"])
            self.assertFalse((output / "model").exists())
            self.assertEqual(dangerous_config.read_text(),
                             "[global]\nindex-url=https://redirect.invalid/simple\n")


if __name__ == "__main__":
    unittest.main()
