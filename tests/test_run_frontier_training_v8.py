"""CPU fixtures only: invented rows/weights, temporary Git and mocked packages."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev.metrics import fit_temperature
from jev.train import read_rows, OPTIMIZER_SETTINGS
from scripts import run_frontier_training_v8 as driver


LIVE_ROOT = driver.ROOT


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False)+'\n')


def row(index, split):
    identity = f'invented-{split}-{index}'
    return {'id': identity, 'group_id': identity, 'source': 'fixture-only', 'split': split,
        'state': {'marker': identity}, 'question': 'Choose the fixture status.', 'kind': 'choice',
        'options': ['active', 'closed'], 'target': [1., 0.],
        'metadata': {'family': 'policy', 'template_id': 'fixture-'+split,
            'entity_ids': [identity], 'provenance': {'type': 'synthetic', 'license': 'CC0-1.0',
                'split_policy': 'whole-parent fixture', 'generator_version': 'fixture-only',
                'seed': 1, 'group_index': index, 'variant': 0}}}


class DriverFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source, self.data, self.checkpoint = [self.base/name for name in ('source', 'data', 'released')]
        for path in (self.source, self.data, self.checkpoint): path.mkdir()
        self.task = self.base/('frontier-v8-'+'a'*16); self.task.mkdir()
        for name in driver.SOURCE_FILES:
            path = self.source/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((LIVE_ROOT/name).read_bytes())
        lock_name = 'requirements-linux-py311-cu128-20261002.lock'
        shutil.copyfile(LIVE_ROOT/lock_name, self.source/lock_name)
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        def git(*args):
            return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'commit.gpgsign=false',
                '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', *args],
                cwd=self.source, env=env, stderr=subprocess.DEVNULL)
        self.git = git
        git('init', '-q'); git('add', '.'); git('commit', '-qm', 'Invented fixture source')
        self.commit = git('rev-parse', 'HEAD').decode().strip()
        for split, count in (('train', 2932), ('calibration', 496)):
            (self.data/(split+'.jsonl')).write_text(''.join(json.dumps(row(i, split))+'\n' for i in range(count)))
        for name in ('validation.jsonl', 'test.jsonl', 'ood.jsonl', *[
            f'observed-regression/{v}/{s}.jsonl' for v in ('v4', 'v5', 'v6', 'v7') for s in ('test', 'ood')]):
            path = self.data/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'OPAQUE INVENTED RESERVED BYTES; NOT JSON\n')
        data_hashes = driver.inventory(self.data)[0]
        write(self.data/'manifest.json', {'files_sha256': {n: h for n, h in data_hashes.items() if '/' not in n},
            'retained_files_sha256': {n: h for n, h in data_hashes.items() if '/' in n}})
        model = dict(model_id=driver.SETTINGS['model'], revision=driver.SETTINGS['revision'],
            method='independent_candidate_lora_nll_brier', lora_rank=8, max_length=4096)
        (self.checkpoint/'adapter').mkdir()
        write(self.checkpoint/'model.json', model)
        write(self.checkpoint/'adapter/adapter_config.json', {'peft_type': 'LORA', 'r': 8})
        write(self.checkpoint/'temperature.json', {'temperature': 1.518796342858676})
        (self.checkpoint/'head.pt').write_bytes(b'Invented CPU fixture, never loaded')
        (self.checkpoint/'adapter/adapter_model.safetensors').write_bytes(b'Invented CPU fixture, never loaded')
        self.cache = self.base/'hf-cache'; self.cache.mkdir()
        self.snapshot = self.cache/'snapshots'/driver.SETTINGS['revision']; self.snapshot.mkdir(parents=True)
        (self.snapshot/'config.json').write_bytes(b'{"fixture":true}\n')
        (self.cache/'blob').write_bytes(b'Invented base weight bytes, never loaded')
        (self.snapshot/'model.safetensors').symlink_to(self.cache/'blob')
        self.packages = self.base/'packages'; self.packages.mkdir()
        self.pins = dict(line.split('==') for line in (self.source/lock_name).read_text().splitlines())
        for name in driver.RUNTIME_NAMES:
            path = self.packages/name; path.mkdir(); (path/'__init__.py').write_text('# invented runtime package\n')
        self.runtime = {name: self.pins[name] for name in driver.RUNTIME_NAMES}
        self.runtime_proof = {'versions': self.pins,
            'distribution_roots': {n: str(self.packages) for n in self.pins},
            'package_origins': {n: str(self.packages/n/'__init__.py') for n in driver.RUNTIME_NAMES}}
        self.plan = {'schema_version': 1, 'protocol_id': driver.PROTOCOL, 'settings': copy.deepcopy(driver.SETTINGS),
            'runtime_settings': copy.deepcopy(driver.NUMERICAL_SETTINGS), 'optimizer': OPTIMIZER_SETTINGS,
            'resource_protocol_ready': False, 'gpu_authority': False, 'data_manifest_sha256': driver.sha(self.data/'manifest.json'),
            'data_files_sha256': data_hashes, 'published_temperature': 1.518796342858676,
            'expected_initial_checkpoint': {'model_repo': 'ZefanCai/Open-Jev-2B',
                'model_revision': '0c7aa498b1627be8da4acf34c863ff0ee0a92785',
                'files_sha256': driver.inventory(self.checkpoint)[0], 'directory_sha256': driver.directory_sha(self.checkpoint)},
            'expected_base_snapshot_files_sha256': driver.inventory(self.snapshot, cache=self.cache)[0],
            'implementation_sha256': {n: driver.sha(self.source/n) for n in driver.SOURCE_FILES},
            'runtime': self.runtime, 'runtime_lock': {'path': lock_name, 'sha256': driver.sha(self.source/lock_name),
                'distributions': 76, 'platform': 'Linux', 'machine': 'x86_64', 'python': '3.11'}}
        self.plan_path, self.freeze_path = self.base/'plan.json', self.base/'freeze.json'
        write(self.plan_path, self.plan)
        self.freeze = {'schema_version': 1, 'status': 'frontier_v8_protocol_frozen_to_committed_source',
            'source_commit': self.commit, 'source_pushed_before_freeze': True,
            'plan_sha256': driver.sha(self.plan_path), 'data_manifest_sha256': self.plan['data_manifest_sha256'],
            'implementation_sha256': self.plan['implementation_sha256']}
        write(self.freeze_path, self.freeze)
        self.stage = {'schema_version': 1, 'status': 'cpu_staged_no_cuda_initialization', 'cuda_initialized': False,
            'protocol_id': driver.PROTOCOL, 'attempt_id': self.task.name, 'source_commit': self.commit,
            'plan_sha256': driver.sha(self.plan_path), 'runtime': self.runtime, 'python': sys.executable,
            'python_sha256': driver.sha(sys.executable), 'runtime_identity': self.runtime_proof,
            'hf_cache': str(self.cache), 'base_snapshot': str(self.snapshot),
            'base_snapshot_files_sha256': self.plan['expected_base_snapshot_files_sha256'],
            'base_snapshot_symlink_targets': driver.inventory(self.snapshot, cache=self.cache)[1]}
        write(self.task/'cpu-stage-receipt.json', self.stage)
        self.request = {'schema_version': 1, 'protocol_id': driver.PROTOCOL, 'attempt_id': self.task.name,
            'node_alias': 'ms-n1-3', 'source_commit': self.commit, 'source_directory': str(self.source),
            'dataset': str(self.data), 'released_checkpoint': str(self.checkpoint), 'task_directory': str(self.task),
            'training_run': str(self.task/'training'), 'comparison_output': str(self.task/'comparison'),
            'completion_receipt': str(self.task/'training-completion.json'),
            'runtime_receipt': str(self.task/'cpu-stage-receipt.json'), 'resource_receipt': str(self.task/'resource-ready.json'),
            'runtime_receipt_sha256': driver.sha(self.task/'cpu-stage-receipt.json'), 'runtime': self.runtime,
            'python': sys.executable, 'hf_cache': str(self.cache), 'base_snapshot': str(self.snapshot)}
        self.request_path = self.task/'execution-request.json'; write(self.request_path, self.request)
        self.args = SimpleNamespace(request=str(self.request_path), plan=str(self.plan_path), freeze=str(self.freeze_path),
            expected_plan_sha256=driver.sha(self.plan_path), expected_freeze_sha256=driver.sha(self.freeze_path),
            expected_commit=self.commit)
        self.stack = [patch.object(driver, 'ROOT', self.source), patch.object(driver.platform, 'system', return_value='Linux'),
            patch.object(driver.platform, 'machine', return_value='x86_64'),
            patch.object(driver.sys, 'version_info', SimpleNamespace(major=3, minor=11)),
            patch.object(driver.metadata, 'distribution', side_effect=lambda n: SimpleNamespace(
                version=self.pins[n], locate_file=lambda _: self.packages)),
            patch.object(driver.util, 'find_spec', side_effect=lambda n: SimpleNamespace(origin=str(self.packages/n/'__init__.py')))]
        for item in self.stack: item.start(); self.addCleanup(item.stop)

    def prepare(self): return driver.prepare(self.args)

    def test_complete_read_only_preflight_and_exact_deferred_command(self):
        before = set(self.task.iterdir()); request, plan, receipt = self.prepare()
        self.assertEqual(before, set(self.task.iterdir()))
        self.assertEqual(receipt['status'], 'cpu_preflight_passed_launch_unavailable')
        self.assertEqual(len(receipt['runtime_identity']['versions']), 76)
        self.assertEqual(receipt['train_rows'], 733*4)
        self.assertEqual(receipt['calibration_rows'], 496)
        argv = receipt['resolved_training_argv']
        self.assertEqual(argv[:3], [sys.executable, '-m', 'jev.train'])
        self.assertEqual(argv[argv.index('--steps')+1], '733')
        self.assertEqual(argv[argv.index('--seed')+1], '20261004')
        self.assertIn('--defer-heldout', argv); self.assertNotIn('--resume-training', argv)
        self.assertFalse(receipt['execution_available']); self.assertFalse(receipt['resource_protocol_ready'])

    def test_reserved_files_are_never_parsed(self):
        from jev import train
        original = train.read_rows
        def guarded(path, *args, **kwargs):
            self.assertIn(Path(path).name, ('train.jsonl', 'calibration.jsonl'))
            return original(path, *args, **kwargs)
        with patch.object(train, 'read_rows', side_effect=guarded): self.prepare()

    def test_dirty_source_and_different_commit_rejected(self):
        path = self.source/'jev/train.py'; old = path.read_bytes(); path.write_bytes(old+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Dirty'): self.prepare()
        self.git('add', '.'); self.git('commit', '-qm', 'Changed fixture')
        with self.assertRaisesRegex(ValueError, 'checkout'): self.prepare()

    def test_git_environment_cannot_redirect_source(self):
        calls = []; actual = driver.subprocess.check_output
        def checked(*args, **kwargs):
            self.assertFalse(any(k.startswith('GIT_') for k in kwargs['env'])); calls.append(args)
            return actual(*args, **kwargs)
        with patch.dict(os.environ, {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.worktree',
            'GIT_CONFIG_VALUE_0': '/foreign', 'GIT_OBJECT_DIRECTORY': '/foreign'}), \
            patch.object(driver.subprocess, 'check_output', side_effect=checked): self.prepare()
        self.assertTrue(calls)

    def test_source_inventory_cannot_claim_other_bytes(self):
        self.plan['implementation_sha256']['jev/train.py'] = '0'*64
        write(self.plan_path, self.plan); self.args.expected_plan_sha256 = driver.sha(self.plan_path)
        self.freeze['plan_sha256'] = self.args.expected_plan_sha256
        self.freeze['implementation_sha256'] = self.plan['implementation_sha256']; write(self.freeze_path, self.freeze)
        self.args.expected_freeze_sha256 = driver.sha(self.freeze_path)
        with self.assertRaisesRegex(ValueError, 'implementation'): self.prepare()

    def test_changed_data_checkpoint_and_base_rejected(self):
        for path in (self.data/'test.jsonl', self.checkpoint/'head.pt', self.cache/'blob'):
            before = path.read_bytes(); path.write_bytes(before+b'changed')
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'dataset|initializer|base content'):
                self.prepare()
            path.write_bytes(before)

    def test_runtime_version_or_position_shadowing_rejected(self):
        self.pins['torch'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'runtime|Runtime'): self.prepare()
        self.pins['torch'] = self.runtime['torch']
        with patch.object(driver.util, 'find_spec', return_value=SimpleNamespace(origin='/foreign/shadow.py')):
            with self.assertRaisesRegex(ValueError, 'stage'): self.prepare()

    def test_runtime_lock_platform_or_stage_mutation_rejected(self):
        with patch.object(driver.platform, 'machine', return_value='arm64'):
            with self.assertRaisesRegex(ValueError, 'platform'): self.prepare()
        (self.task/'cpu-stage-receipt.json').write_bytes(b'changed\n')
        with self.assertRaisesRegex(ValueError, 'staging'): self.prepare()

    def test_symlink_and_special_dataset_leaf_rejected(self):
        p = self.data/'test.jsonl'; p.unlink(); p.symlink_to(self.data/'ood.jsonl')
        with self.assertRaisesRegex(ValueError, 'symlink'): self.prepare()
        p.unlink(); os.mkfifo(p)
        with self.assertRaisesRegex(ValueError, 'Nonregular'): self.prepare()

    def test_special_manifest_or_plan_rejected_before_open(self):
        for path in (self.data/'manifest.json', self.plan_path):
            before = path.read_bytes(); path.unlink(); os.mkfifo(path)
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Regular file'):
                self.prepare()
            path.unlink(); path.write_bytes(before)

    def test_hf_link_must_remain_inside_declared_cache(self):
        p = self.snapshot/'model.safetensors'; p.unlink()
        outside = self.base/'outside'; outside.write_bytes((self.cache/'blob').read_bytes()); p.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'cache'): self.prepare()

    def test_old_attempt_protected_node_placeholder_overlap_and_existing_outputs_refused(self):
        for change in ({'attempt_id': 'boundary-v7-old'}, {'node_alias': 'ms-n1-2'},
                       {'dataset': '/future-host/data'}, {'task_directory': str(self.source)},
                       {'training_run': str(self.data/'training')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                driver.validate_paths({**self.request, **change}, self.request_path)
        (self.task/'old-attempt.lock').touch()
        with self.assertRaisesRegex(ValueError, 'old attempt'): self.prepare()

    def test_changed_plan_or_freeze_bytes_rejected(self):
        for path in (self.plan_path, self.freeze_path):
            before = path.read_bytes(); path.write_bytes(before+b' ')
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'plan/freeze'): self.prepare()
            path.write_bytes(before)

    def test_attempt_is_consumed_even_when_preflight_fails(self):
        (self.data/'test.jsonl').write_bytes(b'changed reserved fixture\n')
        with self.assertRaisesRegex(ValueError, 'dataset'): driver.execute(self.args)
        self.assertTrue((self.task/'attempt.lock.json').is_file())
        self.assertTrue((self.task/'attempt-failure.json').is_file())
        before = (self.task/'attempt.lock.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'consumed'): driver.execute(self.args)
        self.assertEqual(before, (self.task/'attempt.lock.json').read_bytes())

    def test_resource_receipt_flag_cannot_unlock_execution(self):
        write(self.task/'resource-ready.json', {'status': 'ready_for_single_attempt', 'ownership_verified': True})
        actual_run = driver.subprocess.run
        def guarded(command, **kwargs):
            self.assertEqual(command[0], 'git', 'Only CPU Git checks may execute')
            return actual_run(command, **kwargs)
        with patch.object(driver.subprocess, 'run', side_effect=guarded):
            with self.assertRaisesRegex(ValueError, 'Execution unavailable'): driver.execute(self.args)
        self.assertFalse((self.task/'training').exists())

    def completed_fixture(self):
        _, _, preflight = self.prepare(); run = self.task/'training'; run.mkdir()
        rows = read_rows(self.data/'calibration.jsonl', 0, driver.SETTINGS['seed'], balanced=True)
        predictions = [{k: r[k] for k in ('id', 'kind', 'source', 'group_id', 'target')} for r in rows]
        for r in predictions: r.update(question_id='choice', target_basis='hard_label', logits=[1., -1.],
            probabilities=[.880797, .119203], latency_seconds=.01)
        self.assertTrue(all('options' not in r for r in predictions))  # Exact trainer evaluate shape.
        (run/'calibration.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        (run/'reload_check.jsonl').write_text(json.dumps(predictions[0])+'\n')
        phases = {n: {'elapsed_seconds': 1., 'peak_allocated_tensor_memory_gib': 2.,
            'peak_reserved_allocator_memory_gib': 3.} for n in ['model_loading', 'warmup', 'optimizer',
            'checkpoint_save', 'calibration_and_temperature', 'checkpoint_reload']}
        heldout = {'status': 'deferred', 'splits': ['validation', 'test', 'ood'], 'model_calls': 0}
        meta = {**driver.SETTINGS, 'commit': self.commit, 'resume_training': None, 'phase': 'complete',
            'initial_checkpoint_identity': preflight['initial_checkpoint'], 'baseline_initialization': 'inference_checkpoint',
            'heldout_evaluation': heldout, 'evaluation_ids': [], 'ood_ids': [],
            'data_sha256': {s: self.plan['data_files_sha256'][s+'.jsonl'] for s in ('train', 'calibration')},
            'calibration_ids': [r['id'] for r in rows], 'phase_metrics': phases}
        temperature = fit_temperature([r['logits'] for r in predictions], [r['target'] for r in predictions])
        summary = {'status': 'complete', 'steps': 733, 'trained_rows_consumed': 2932,
            'checkpoint_selection': 'fixed_final_step', 'checkpoint_reload_split': 'calibration',
            'metrics': {}, 'baseline_temperature': None, 'heldout_evaluation': heldout,
            'temperature': temperature, 'checkpoint_reload_max_error': 0., 'phase_metrics': phases}
        write(run/'run.json', meta); write(run/'summary.json', summary)
        (run/'training.jsonl').write_text(''.join(json.dumps({'step': i, 'loss': .1, 'gradient_norm': .1,
            'elapsed_seconds': float(i), 'peak_memory_gib': 2.})+'\n' for i in range(1, 734)))
        shutil.copytree(self.checkpoint, run/'checkpoint')
        write(run/'checkpoint/temperature.json', {'temperature': temperature, 'split': 'calibration', 'n': 496,
            'ids_sha256': hashlib.sha256(json.dumps(meta['calibration_ids']).encode()).hexdigest()})
        return run, preflight, meta, summary

    def test_real_trainer_journal_shape_without_options_is_supported(self):
        run, preflight, _, _ = self.completed_fixture()
        receipt = driver.validate_completed_run(run, self.plan, preflight)
        self.assertEqual(receipt['completed_steps'], 733); self.assertEqual(receipt['consumed_rows'], 2932)
        self.assertEqual(receipt['calibration']['count'], 496)

    def test_incomplete_nonfinite_or_nonmonotonic_optimizer_steps_rejected(self):
        run, preflight, _, _ = self.completed_fixture(); p = run/'training.jsonl'
        records = [json.loads(line) for line in p.read_bytes().splitlines()]
        for bad in (records[:-1], list(reversed(records)), [{**records[0], 'loss': math.nan}, *records[1:]],
                    [records[0], {**records[1], 'elapsed_seconds': 0.}, *records[2:]]):
            p.write_text(''.join(json.dumps(r)+'\n' for r in bad))
            with self.assertRaisesRegex(ValueError, 'journal|elapsed'): driver.validate_completed_run(run, self.plan, preflight)

    def test_changed_calibration_labels_initial_identity_reload_or_phase_rejected(self):
        run, preflight, meta, summary = self.completed_fixture()
        for mode in ('initial', 'reload', 'phase', 'heldout'):
            changed = copy.deepcopy(meta); altered = copy.deepcopy(summary)
            if mode == 'initial': changed['initial_checkpoint_identity']['temperature'] += 1
            if mode == 'reload': altered['checkpoint_reload_max_error'] = .01
            if mode == 'phase': altered['phase_metrics']['warmup']['elapsed_seconds'] = -1
            if mode == 'heldout': changed['evaluation_ids'] = ['invented-forbidden']
            write(run/'run.json', changed); write(run/'summary.json', altered)
            with self.subTest(mode=mode), self.assertRaises(ValueError): driver.validate_completed_run(run, self.plan, preflight)
        write(run/'run.json', meta); write(run/'summary.json', summary)
        p = run/'calibration.jsonl'; records = [json.loads(line) for line in p.read_bytes().splitlines()]
        records[0]['target'] = [0., 1.]; p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        with self.assertRaisesRegex(ValueError, 'Calibration logits'): driver.validate_completed_run(run, self.plan, preflight)

    def test_incomplete_final_checkpoint_and_special_journal_rejected(self):
        run, preflight, _, _ = self.completed_fixture()
        (run/'checkpoint/head.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'Initial checkpoint'): driver.validate_completed_run(run, self.plan, preflight)
        shutil.copyfile(self.checkpoint/'head.pt', run/'checkpoint/head.pt')
        path = run/'reload_check.jsonl'; path.unlink(); os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, 'Nonregular'): driver.validate_completed_run(run, self.plan, preflight)


if __name__ == '__main__':
    unittest.main()
