"""CPU contracts: immutable staging, path translation, one attempt and GPU proof."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_boundary_training_v7 as driver
from scripts import compare_boundary_training_v7 as comparison
from scripts import compare_policy_training_v6 as helpers


def write(path, value):
    Path(path).write_text(json.dumps(value)+'\n')


class PathAndAttemptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.task = Path(self.temporary.name)
        self.request = {name: str(self.task/name) for name in driver.PATHS}
        self.request['source_directory'] = str(driver.ROOT)

    def test_translation_changes_only_three_paths_and_preserves_frozen_plan(self):
        plan = json.loads(driver.PLAN.read_text()); original = copy.deepcopy(plan)
        argv = driver.translated_argv(plan, self.request)
        expected = list(plan['training_argv'][3:])
        for flag, name in (('--data', 'dataset'), ('--output', 'training_run'), ('--initial-checkpoint', 'released_checkpoint')):
            expected[expected.index(flag)+1] = self.request[name]
        self.assertEqual(argv, expected)
        self.assertEqual(plan, original)
        self.assertNotIn('--resume-training', argv)
        self.assertEqual(argv[argv.index('--steps')+1], '948')
        self.request['training_run'] = '/future-host/run'
        with self.assertRaisesRegex(ValueError, 'future-host'):
            driver.translated_argv(plan, self.request)

    def test_refuse_existing_output_nested_input_parent_and_placeholder(self):
        driver.validate_paths(self.request)
        bad = dict(self.request, training_run=str(Path(self.request['dataset'])/'adaptation'))
        with self.assertRaisesRegex(ValueError, 'overlaps immutable'):
            driver.validate_paths(bad)
        bad = dict(self.request, comparison_output=str(Path(self.request['training_run'])/'comparison'))
        with self.assertRaisesRegex(ValueError, 'output paths overlap'):
            driver.validate_paths(bad)
        for output in ('training_run', 'comparison_output', 'completion_receipt'):
            p = Path(self.request[output]);p.touch()
            with self.assertRaisesRegex(ValueError, 'existing attempt'):
                driver.validate_paths(self.request)
            p.unlink()
        with self.assertRaisesRegex(ValueError, 'host path'):
            driver.validate_paths(dict(self.request, released_checkpoint='/future-host/released'))

    def test_attempt_receipt_cannot_be_overwritten(self):
        path = self.task/'attempt.lock.json'
        driver.write_once(path, {'status': 'started'})
        with self.assertRaises(FileExistsError):
            driver.write_once(path, {'status': 'replacement'})
        self.assertEqual(json.loads(path.read_text()), {'status': 'started'})

    def test_duplicate_driver_refusal_does_not_relabel_prior_attempt_as_failure(self):
        request_path = self.task/'execution-request.json';write(request_path, self.request)
        driver.write_once(self.task/'attempt.lock.json', {'status': 'complete'})
        with patch.object(driver.sys, 'argv', ['driver', '--request', str(request_path), '--expected-commit', 'e'*40]), \
             patch.object(driver, 'execute', side_effect=ValueError('Existing attempt')):
            with self.assertRaisesRegex(ValueError, 'Existing attempt'):
                driver.main()
        self.assertFalse((self.task/'attempt-failure.json').exists())

    def test_complete_steps_and_finite_fields_required_without_resume_snapshot(self):
        write(self.task/'summary.json', dict(status='complete', steps=948, trained_rows_consumed=3792))
        journal = [dict(step=i, loss=.1, gradient_norm=.1, elapsed_seconds=float(i), peak_memory_gib=5.) for i in range(1, 949)]
        def save(values):
            (self.task/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
        save(journal);driver.validate_completed_run(self.task)
        for values in (journal[:-1], journal+[journal[-1]], list(reversed(journal))):
            save(values)
            with self.assertRaisesRegex(ValueError, 'journal'):
                driver.validate_completed_run(self.task)
        for name in ('loss', 'gradient_norm', 'elapsed_seconds', 'peak_memory_gib'):
            for value in (math.nan, -1.):
                bad = copy.deepcopy(journal);bad[0][name] = value;save(bad)
                with self.assertRaisesRegex(ValueError, 'journal field'):
                    driver.validate_completed_run(self.task)
        save(journal);(self.task/'training-checkpoints').mkdir()
        with self.assertRaisesRegex(ValueError, 'resumable'):
            driver.validate_completed_run(self.task)


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.task = Path(self.temporary.name)
        source_data = driver.ROOT/'data/boundary-training-v7-20261002-r1'
        # Public CI has manifests/locks but not ignored prepared data. Preparation tests cover regeneration.
        if not source_data.exists():
            self.skipTest('Frozen local mixture is not shipped in portable CI')
        data = self.task/'data';shutil.copytree(source_data, data)
        plan = json.loads(driver.PLAN.read_text())
        self.live_source = driver.ROOT
        self.source = self.task/'frozen-source'
        self.evaluation_source = self.task/'evaluation-source'
        environment = {name: value for name, value in os.environ.items() if not name.startswith('GIT_')}
        # A newer trainer must not change the positive fixture's frozen v7 implementation.
        for name in sorted({*plan['implementation_sha256'], 'jev/model.py'}):
            raw = subprocess.check_output(
                ['git', '-C', str(self.live_source), 'show', driver.SOURCE_COMMIT+':'+name],
                env=environment, stderr=subprocess.DEVNULL)
            digest = hashlib.sha256(raw).hexdigest()
            if name in plan['implementation_sha256']:
                self.assertEqual(digest, plan['implementation_sha256'][name])
            for checkout in (self.source, self.evaluation_source):
                path = checkout/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(raw)
                self.assertEqual(driver.sha(path), digest)
        runtime = dict(zip(driver.RUNTIME_NAMES, ('2.8.0', '5.10.2', '0.19.1', '3.4.0', '0.7.0', '1.13.0')))
        revision = '15852e8c16360a2fea060d615a32b45270f8a8fc'
        snapshot = self.task/'cache'/'models--Qwen--Qwen3.5-2B'/'snapshots'/revision
        snapshot.mkdir(parents=True);(snapshot/'model.safetensors').write_bytes(b'CPU fixture, not weights')
        stage = dict(status='cpu_staged_no_cuda_initialization', cuda_initialized=False,
            source_commit=driver.SOURCE_COMMIT, runtime=runtime, base_snapshot=str(snapshot),
            base_snapshot_files_sha256={'model.safetensors': driver.sha(snapshot/'model.safetensors')})
        write(self.task/'cpu-stage-receipt.json', stage)
        self.request = dict(schema_version=1, evaluation_commit='e'*40, plan_sha256=driver.PLAN_SHA256,
            source_directory=str(self.source), dataset=str(data), released_checkpoint=str(self.task/'released'),
            training_run=str(self.task/'adaptation'), comparison_output=str(self.task/'comparison'),
            completion_receipt=str(self.task/'completion-receipt.json'), resource_receipt=str(self.task/'resource-ready.json'),
            cpu_stage_receipt_sha256=driver.sha(self.task/'cpu-stage-receipt.json'), runtime=runtime,
            gpu_uuid='GPU-8e9ce19f-1174-0848-8e17-3201e3bb8775')
        self.args = SimpleNamespace(request=str(self.task/'execution-request.json'), expected_commit='e'*40)
        write(self.args.request, self.request)
        initial = dict(files_sha256=plan['expected_initial_checkpoint']['files_sha256'], config=dict(revision=revision))
        for patcher in (patch.object(driver, 'ROOT', self.evaluation_source),
                        patch.object(comparison, 'ROOT', self.evaluation_source),
                        patch.object(comparison, 'source_identity', return_value={}),
                        patch.object(driver, 'git', side_effect=lambda path, *args: driver.SOURCE_COMMIT if args[0] == 'rev-parse' else ''),
                        patch.object(driver, 'version', side_effect=runtime.__getitem__),
                        patch.object(helpers, 'checkpoint_identity', return_value=initial),
                        patch.object(helpers, 'directory_sha', return_value=plan['expected_initial_checkpoint']['directory_sha256'])):
            patcher.start();self.addCleanup(patcher.stop)
        self.stage = stage

    def test_real_frozen_data_preflight_without_gpu_acquisition(self):
        request, plan, receipt = driver.prepare(self.args)
        self.assertEqual(receipt['configured_train_rows'], 3792)
        self.assertEqual(len(receipt['data_files_sha256']), 19)
        self.assertFalse(receipt['direct_per_row_consumption_observed'])
        self.assertFalse((self.task/'attempt.lock.json').exists())
        self.assertFalse(Path(request['resource_receipt']).exists())

    def test_data_runtime_stage_and_base_tampering_refused_before_model(self):
        for path in (Path(self.request['dataset'])/'test.jsonl', self.task/'cpu-stage-receipt.json',
                     Path(self.stage['base_snapshot'])/'model.safetensors'):
            old = path.read_bytes();path.write_bytes(old+b'changed')
            with self.subTest(path=path), self.assertRaises(ValueError):
                driver.prepare(self.args)
            path.write_bytes(old)
        write(self.args.request, {**self.request, 'runtime': {**self.request['runtime'], 'peft': 'changed'}})
        with self.assertRaisesRegex(ValueError, 'package versions'):
            driver.prepare(self.args)

    def test_old_attempt_cannot_reenter_preflight(self):
        write(self.task/'attempt.lock.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'already started'):
            driver.prepare(self.args)

    def test_edited_or_live_trainer_refused_before_data_checkpoint_and_runtime_staging(self):
        frozen = (self.source/'jev/train.py').read_bytes()
        changed = [('edited', frozen+b'\n# Deliberately changed CPU fixture trainer.\n')]
        live = (self.live_source/'jev/train.py').read_bytes()
        if live != frozen:
            changed.append(('live', live))
        actual_open = Path.open
        forbidden = (Path(self.request['dataset']), Path(self.stage['base_snapshot']))
        def guarded_open(path, *args, **kwargs):
            candidate = Path(path)
            if any(candidate.is_relative_to(root) for root in forbidden) or candidate.name == 'cpu-stage-receipt.json':
                raise AssertionError('Frozen source refusal must precede data/checkpoint/runtime stage reads')
            return actual_open(path, *args, **kwargs)
        for checkout in (self.source, self.evaluation_source):
            path = checkout/'jev/train.py'
            for label, raw in changed:
                with self.subTest(checkout=checkout.name, source=label):
                    path.write_bytes(raw)
                    try:
                        with patch.object(Path, 'open', guarded_open), \
                             patch.object(helpers, 'checkpoint_identity', side_effect=AssertionError('Checkpoint stage reached')):
                            with self.assertRaisesRegex(ValueError, 'Frozen implementation changed: jev/train.py'):
                                driver.prepare(self.args)
                    finally:
                        path.write_bytes(frozen)
                    self.assertFalse((self.task/'attempt.lock.json').exists())
                    self.assertFalse(Path(self.request['training_run']).exists())


class ResourceProofTests(unittest.TestCase):
    def test_stale_proof_guard_birth_compute_process_and_busy_gpu_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory)/'resource.json'
            boot = 'fixture-boot';uuid = 'GPU-8e9ce19f-1174-0848-8e17-3201e3bb8775'
            request = dict(resource_receipt=str(receipt_path), gpu_uuid=uuid)
            receipt = dict(status='ready_for_single_attempt', ownership_verified=True, gpu_uuid=uuid, boot_id=boot,
                checked_at_utc=datetime.now(timezone.utc).isoformat(), restoration_required=True,
                restoration_guard_identity=dict(pid=12345, uid=os.getuid(), start_ticks=77, boot_id=boot))
            real_read = Path.read_text;guard_state = ['S']
            def read(path, *args, **kwargs):
                if str(path) == '/proc/sys/kernel/random/boot_id':return boot
                if str(path) == '/proc/12345/stat':return '12345 (fixture guard) '+' '.join([guard_state[0]]+['0']*18+['77'])
                return real_read(path, *args, **kwargs)
            def stat(path, *args, **kwargs):
                if str(path) == '/proc/12345':return SimpleNamespace(st_uid=os.getuid())
                return real_stat(path, *args, **kwargs)
            real_stat = Path.stat
            with patch.object(Path, 'read_text', read), patch.object(Path, 'stat', stat), \
                 patch.dict(os.environ, CUDA_VISIBLE_DEVICES=uuid), patch.object(driver.subprocess, 'check_output') as probe:
                write(receipt_path, receipt);probe.side_effect = ['', uuid+', 4, 0\n']
                driver.validate_resource(request)
                for change in (dict(checked_at_utc=(datetime.now(timezone.utc)-timedelta(minutes=3)).isoformat()),
                               dict(ownership_verified=False), dict(boot_id='different'),
                               dict(restoration_guard_identity={**receipt['restoration_guard_identity'], 'start_ticks': 78})):
                    write(receipt_path, {**receipt, **change})
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        driver.validate_resource(request)
                write(receipt_path, receipt)
                for state in ('Z', 'X', 'T', 't'):
                    guard_state[0] = state
                    with self.assertRaisesRegex(ValueError, 'guard identity'):
                        driver.validate_resource(request)
                guard_state[0] = 'S'
                probe.side_effect = ['555\n']
                with self.assertRaisesRegex(ValueError, 'compute processes'):
                    driver.validate_resource(request)
                for memory, utilization in ((257, 0), (4, 1)):
                    probe.side_effect = ['', f'{uuid}, {memory}, {utilization}\n']
                    with self.assertRaisesRegex(ValueError, 'idle memory'):
                        driver.validate_resource(request)


if __name__ == '__main__':
    unittest.main()
