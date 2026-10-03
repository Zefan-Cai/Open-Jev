"""Independent hand-authored CPU replay arithmetic and provenance guards."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import replay_frontier_comparison_v8 as replay


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/frontier-v8-training-protocol-20261003/training-plan.json'
MEMORY = {'allocated_baseline_bytes': 100, 'reserved_baseline_bytes': 150,
          'allocated_peak_bytes': 200, 'reserved_peak_bytes': 250, 'scope': replay.MEMORY_SCOPE}


def row(identity='hand/one', kind='noul', truth='yes', split='test'):
    options = ['no', 'yes'] if kind == 'noul' else ['closed', 'active', 'unknown']
    return {'id': identity, 'group_id': identity+'/parent', 'source': 'hand-authored-replay-fixture',
        'split': split, 'state': {'account': 'contract account', 'events': [
            {'account': 'contract account', 'sequence': 2, 'status': 'closed'},
            {'account': 'other account', 'sequence': 100, 'status': 'active'}]},
        'question': "Does the recorded rule establish 'closed' for this case?" if kind == 'noul' else 'Choose the current account status.',
        'kind': kind, 'options': options, 'target': [float(v == truth) for v in options],
        'metadata': {'scenario_family': 'state_tracking', 'condition': 'closed_last', 'layout': 'reverse'}}


def prediction(value, index=0, logits=None):
    return {**{k: copy.deepcopy(value[k]) for k in ('id', 'group_id', 'source', 'kind', 'options', 'target')},
        'index': index, 'slice': value['split'], 'weight': 'released', 'status': 'complete',
        'row_sha256': replay.digest(value), 'checkpoint_directory_sha256': 'a'*64,
        'logits': logits or [0., 2.], 'latency_seconds': .01,
        'timing_scope': replay.TIMING_SCOPE, 'cuda_memory': dict(MEMORY)}


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def save_journal(path, records):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))


def completed_fixture(directory):
    """A complete mock optimizer/Calibration receipt with no model execution."""
    plan = json.loads(PLAN.read_text()); root = Path(directory)
    run = root/'training'; (run/'checkpoint').mkdir(parents=True)
    calibration = [row('cal/独立/'+str(i), split='calibration') for i in range(496)]
    ordered = replay.calibration_order(calibration, plan['settings']['seed'])
    ids = [r['id'] for r in ordered]
    records = [{**{k: r[k] for k in ('id', 'group_id', 'source', 'kind', 'target')}, 'logits': [0., 2.]} for r in ordered]
    temperature = replay.fit_calibration_temperature(ordered, records)
    initial = {'files_sha256': {'released': 'f'*64}, 'temperature': plan['published_temperature']}
    preflight = {'source_commit': 'c'*40, 'plan_sha256': 'd'*64, 'freeze_receipt_sha256': 'e'*64,
        'initial_checkpoint': initial, 'calibration_selection_sha256': replay.training_digest(ids)}
    phases = {k: {'elapsed_seconds': 1., 'peak_allocated_tensor_memory_gib': 2.,
        'peak_reserved_allocator_memory_gib': 3.} for k in ('model_loading', 'warmup', 'optimizer',
        'checkpoint_save', 'calibration_and_temperature', 'checkpoint_reload')}
    deferred = {'status': 'deferred', 'splits': ['validation', 'test', 'ood'], 'model_calls': 0}
    meta = {**plan['settings'], 'phase': 'complete', 'commit': 'c'*40, 'resume_training': None,
        'initial_checkpoint_identity': initial, 'baseline_initialization': 'inference_checkpoint',
        'heldout_evaluation': deferred, 'evaluation_ids': [], 'ood_ids': [],
        'data_sha256': {s: plan['data_files_sha256'][s+'.jsonl'] for s in ('train', 'calibration')},
        'calibration_ids': ids, 'phase_metrics': phases}
    summary = {'status': 'complete', 'steps': 733, 'trained_rows_consumed': 2932,
        'checkpoint_selection': 'fixed_final_step', 'checkpoint_reload_split': 'calibration',
        'metrics': {}, 'baseline_temperature': None, 'heldout_evaluation': deferred,
        'temperature': temperature, 'checkpoint_reload_max_error': 0., 'phase_metrics': phases}
    save(run/'run.json', meta); save(run/'summary.json', summary)
    save_journal(run/'training.jsonl', [{'step': i, 'loss': .1, 'gradient_norm': 1.,
        'elapsed_seconds': float(i), 'peak_memory_gib': 2.} for i in range(1, 734)])
    save_journal(run/'calibration.jsonl', records); save_journal(run/'reload_check.jsonl', [records[0]])
    save(run/'checkpoint/temperature.json', {'split': 'calibration', 'n': 496,
        'ids_sha256': replay.raw_sha(json.dumps(ids).encode()), 'temperature': temperature})
    checkpoint = {'files_sha256': replay.inventory(run/'checkpoint')[0], 'directory_sha256': replay.directory_digest(run/'checkpoint')}
    completion = {'schema_version': 1, 'status': 'complete', 'protocol_id': replay.PROTOCOL,
        'source_commit': 'c'*40, 'plan_sha256': 'd'*64, 'freeze_receipt_sha256': 'e'*64,
        'runtime': plan['runtime'], 'completed_steps': 733, 'consumed_rows': 2932,
        'fixed_final_checkpoint': True, 'defer_heldout': True, 'resume_training': False,
        'automatic_promotion': False, 'checkpoint': checkpoint, 'training_run': str(run),
        'artifacts_sha256': {n: replay.sha(run/n) for n in ('run.json', 'summary.json', 'training.jsonl', 'calibration.jsonl', 'reload_check.jsonl')},
        'calibration': {'journal_sha256': replay.sha(run/'calibration.jsonl'), 'ordered_ids_sha256': replay.training_digest(ids), 'count': 496, 'temperature': temperature}}
    return {'adapted_checkpoint': str(run/'checkpoint')}, plan, preflight, completion, calibration, {'adapted': checkpoint}


class ReplayTests(unittest.TestCase):
    def test_inclusive_predicate_and_ties(self):
        self.assertEqual([replay.predicate(p) for p in (0., .2, .2000001, .7999999, .8, 1.)],
            ['no', 'no', 'abstained', 'abstained', 'yes', 'yes'])
        value = row(kind='choice', truth='closed')
        self.assertEqual(replay.selected(value, prediction(value, logits=[0., 0., 0.])), 'closed')
        for probability in (-.1, 1.1, float('nan'), True):
            with self.subTest(probability=probability), self.assertRaises(ValueError): replay.predicate(probability)

    def test_known_brier_nll_ece_and_empty(self):
        yes = row('one'); no = row('two', truth='no'); no['question'] = "Does the recorded rule establish 'active' for this case?"
        result = replay.metrics([yes, no], [prediction(yes, logits=[0., 0.]), prediction(no, logits=[0., 0.])], 1.)
        self.assertEqual(result['correct'], 1); self.assertEqual(result['brier'], .5)
        self.assertAlmostEqual(result['nll'], math.log(2)); self.assertEqual(result['multiclass_ece'], 0.)
        self.assertEqual(result['noul']['accepted'], 0); self.assertIsNone(result['noul']['selective_accuracy'])
        self.assertIsNone(replay.metrics([], [], 1.)['nll'])

    def test_calibration_fit_has_known_optimum_and_split_guard(self):
        rows = [row(str(i), truth='yes' if i < 3 else 'no', split='calibration') for i in range(4)]
        rows[-1]['question'] = "Does the recorded rule establish 'active' for this case?"
        values = [prediction(r, logits=[0., math.log(3.)]) for r in rows]
        self.assertAlmostEqual(replay.fit_calibration_temperature(rows, values), 1., places=4)
        rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Calibration-only'): replay.fit_calibration_temperature(rows, values)

    def test_alignment_detects_content_sequence_logits_and_memory(self):
        value = row(); good = prediction(value)
        replay.aligned([value], [good], 'released', 'test', 'a'*64)
        changes = {'index': True, 'id': 'other', 'row_sha256': 'b'*64,
            'checkpoint_directory_sha256': 'b'*64, 'slice': 'ood', 'weight': 'adapted',
            'status': 'pending', 'logits': [0., float('nan')], 'latency_seconds': -.1,
            'timing_scope': 'HTTP', 'target': [False, True]}
        for key, bad in changes.items():
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                replay.aligned([value], [{**good, key: bad}], 'released', 'test', 'a'*64)
        for bad in ([0.], [0., 1., 2.], [float('inf'), 0.]):
            with self.subTest(logits=bad), self.assertRaises(ValueError):
                replay.aligned([value], [{**good, 'logits': bad}], 'released', 'test', 'a'*64)
        bad = copy.deepcopy(good); bad['cuda_memory']['reserved_baseline_bytes'] = 99
        with self.assertRaises(ValueError): replay.aligned([value], [bad], 'released', 'test', 'a'*64)
        with self.assertRaises(ValueError): replay.aligned([value, value], [good, good], 'released', 'test', 'a'*64)

    def test_paired_regressions_fixes_and_threshold_changes(self):
        values = [row(str(i)) for i in range(4)]
        before = [prediction(r, logits=l) for r, l in zip(values, ([0., 4.], [0., 4.], [4., 0.], [4., 0.]))]
        after = [prediction(r, logits=l) for r, l in zip(values, ([0., 4.], [4., 0.], [0., 4.], [4., 0.]))]
        result = replay.paired(values, before, after, 1., 1.)
        self.assertEqual([result[k] for k in ('correct_to_correct', 'correct_to_wrong', 'wrong_to_correct', 'wrong_to_wrong')], [1]*4)
        self.assertEqual(result['regression_ids'], ['1']); self.assertEqual(result['fix_ids'], ['2'])
        self.assertEqual(result['threshold_decision_changes'], 2)

    def test_timing_keeps_cold_outlier_and_nearest_rank(self):
        values = [prediction(row(str(i))) for i in range(20)]
        for i, value in enumerate(values): value['latency_seconds'] = float(i+1)
        values[-1]['latency_seconds'] = 1000.
        result = replay.efficiency(values)
        self.assertEqual(result['median_seconds'], 10.5); self.assertEqual(result['p95_seconds'], 19.)
        self.assertEqual(result['max_seconds'], 1000.); self.assertEqual(result['mean_seconds'], 59.5)
        self.assertEqual(result['reserved_peak_bytes'], 250); self.assertEqual(len(result['all_seconds']), 20)

    def test_strict_json_nonfinite_and_duplicate_keys(self):
        for raw in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e1000}', '{"x":1,"x":2}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): replay.parse(raw)
        self.assertEqual(replay.parse('{"text":"独立"}'), {'text': '独立'})
        self.assertNotEqual(replay.digest(['独立']), replay.training_digest(['独立']))

    def test_inventory_rejects_symlink_fifo_and_unexpected_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'safe').write_text('fixed')
            (root/'link').symlink_to(root/'safe')
            with self.assertRaises(ValueError): replay.inventory(root)
            (root/'link').unlink(); os.mkfifo(root/'pipe')
            with self.assertRaises(ValueError): replay.inventory(root)

    def test_source_freeze_rejects_dirty_tree_and_scrubs_git_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in replay.SOURCE_REQUIRED:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# hand-authored frozen source\n')
            def git(*args): return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)
            git('init', '-q'); git('add', '.')
            git('-c', 'user.name=CPU fixture', '-c', 'user.email=cpu@example.test', 'commit', '-qm', 'Freeze mock source')
            commit = git('rev-parse', 'HEAD').decode().strip(); hashes = replay.inventory(root)[0]
            hashes = {name: checksum for name, checksum in hashes.items() if not name.startswith('.git/')}
            plan = {'implementation_sha256': hashes}; freeze = {'source_commit': commit, 'implementation_sha256': hashes}
            with patch.dict(os.environ, {'GIT_DIR': '/nonexistent/untrusted', 'GIT_WORK_TREE': '/nonexistent'}):
                self.assertEqual(replay.source_binding(plan, freeze, commit, root), hashes)
            (root/'jev/train.py').write_text('# changed\n')
            with self.assertRaises(ValueError): replay.source_binding(plan, freeze, commit, root)

    def test_plan_rejects_changed_safety_temperature_runtime_and_timing(self):
        good = json.loads(PLAN.read_text()); replay.validate_plan(good)
        paths = [(('comparison', 'temperature_strategies'), ['published']),
            (('comparison', 'efficiency', 'p95'), 'linear interpolation'),
            (('comparison', 'safety', 'each_family_pooled_noul_min_coverage'), .4),
            (('runtime_settings', 'tf32'), 0), (('published_temperature',), 1.)]
        for keys, value in paths:
            bad = copy.deepcopy(good); target = bad
            for key in keys[:-1]: target = target[key]
            target[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError): replay.validate_plan(bad)

    def test_completed_training_and_unicode_calibration_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = completed_fixture(temporary)
            replay.training_completion(*fixture)
            bad = copy.deepcopy(fixture[3]); bad['completed_steps'] = 732
            with self.assertRaises(ValueError): replay.training_completion(fixture[0], fixture[1], fixture[2], bad, fixture[4], fixture[5])
            (Path(temporary)/'training/baseline_test.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Unexpected training'): replay.training_completion(*fixture)

    def test_completed_training_rejects_claimed_error_and_nonmonotonic_steps(self):
        for mode in ('reload', 'step'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = completed_fixture(temporary); run = Path(temporary)/'training'
                filename = 'summary.json' if mode == 'reload' else 'training.jsonl'
                if mode == 'reload':
                    summary = replay.read(run/filename); summary['checkpoint_reload_max_error'] = .01; save(run/filename, summary)
                else:
                    records = replay.journal(run/filename); records[10]['elapsed_seconds'] = 0.; save_journal(run/filename, records)
                fixture[3]['artifacts_sha256'][filename] = replay.sha(run/filename)
                with self.assertRaises(ValueError): replay.training_completion(*fixture)

    def test_checkpoint_base_and_adapter_contract(self):
        plan = json.loads(PLAN.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary)/name for name in ('released', 'adapted')]
            model = {'model_id': plan['settings']['model'], 'revision': plan['settings']['revision'],
                'max_length': 4096, 'lora_rank': 8, 'method': 'independent_candidate_lora_nll_brier'}
            adapter = {'peft_type': 'LORA', 'r': 8, 'base_model_name_or_path': plan['settings']['model'],
                'revision': None, 'target_modules': ['q_proj', 'v_proj']}
            for path in paths:
                (path/'adapter').mkdir(parents=True); save(path/'model.json', model); save(path/'adapter/adapter_config.json', adapter)
            replay.checkpoint_contract(*paths, plan, '/mock/base')
            adapter['target_modules'] = ['q_proj']; save(paths[1]/'adapter/adapter_config.json', adapter)
            with self.assertRaises(ValueError): replay.checkpoint_contract(*paths, plan, '/mock/base')

    def test_failed_exclusive_replay_remains_consumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root/'replay.json'; request_path = root/'request.json'
            request = {key: str(root/(key+'.input')) for key in ('dataset', 'released_checkpoint', 'adapted_checkpoint',
                'training_completion_receipt', 'training_preflight_receipt', 'training_execution_request', 'runtime_receipt', 'resource_receipt')}
            save(request['training_completion_receipt'], {'training_run': str(root/'training')}); save(request_path, request)
            args = SimpleNamespace(output=str(output), request=str(request_path), plan=str(root/'plan.json'),
                freeze=str(root/'freeze.json'), comparison=str(root/'comparison'))
            with patch.object(replay, 'provenance', side_effect=ValueError('hand-authored mock failure')):
                with self.assertRaisesRegex(ValueError, 'mock failure'): replay.replay(args)
            self.assertTrue((root/'replay.json.consumed.json').is_file())
            self.assertTrue((root/'replay.json.failure.json').is_file()); self.assertFalse(output.exists())
            with self.assertRaises(FileExistsError): replay.replay(args)

    def test_no_model_or_comparator_import_in_independent_replay(self):
        subprocess.check_call(['python3', '-c', 'import sys; import scripts.replay_frontier_comparison_v8; '
            'assert "torch" not in sys.modules; assert "jev.metrics" not in sys.modules; '
            'assert "scripts.compare_frontier_training_v8" not in sys.modules'], cwd=ROOT)


if __name__ == '__main__': unittest.main()
