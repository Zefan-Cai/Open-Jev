"""Hand-authored CPU contracts, without model calls or reserved dataset rows."""
from contextlib import contextmanager, ExitStack
import copy
import io
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import compare_frontier_training_v8 as comparison


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/frontier-v8-training-protocol-20261003/training-plan.json'
CHECKPOINTS = {'released': 'a'*64, 'adapted': 'b'*64}
MEMORY = {'allocated_baseline_bytes': 100, 'reserved_baseline_bytes': 200,
          'allocated_peak_bytes': 150, 'reserved_peak_bytes': 250, 'scope': comparison.MEMORY_SCOPE}


def row(identity, split, family='temporal_window', kind='choice', truth=None):
    options = ['no', 'yes'] if kind == 'noul' else ['reject', comparison.PERMISSIVE.get(family, 'equal')]
    if family == 'exact_numeric' and kind == 'choice':
        options = ['below', 'equal', 'above']
    truth = truth if truth is not None else options[0]
    return {'id': identity, 'group_id': identity+'/group', 'source': 'hand-authored-test-fixture',
            'split': split, 'state': {'fixture_only': True}, 'question': 'Hand-authored contract example?',
            'kind': kind, 'options': options, 'target': [float(v == truth) for v in options],
            'metadata': {'scenario_family': family, 'condition': 'scope_first_currency',
                         'composition_width': 5, 'layout': 'reverse'}}


def fixtures():
    result = {}
    for name, count in comparison.COUNTS.items():
        split = name.rsplit('_', 1)[1]
        values = []
        if name.startswith('v8'):
            for family, n in comparison.FAMILY_ROWS.items():
                for kind in ('choice', 'noul'):
                    for i in range(n//2):
                        truth = ('no' if i % 2 == 0 else 'yes') if kind == 'noul' else None
                        values.append(row(name+'/'+family+'/'+kind+'/'+str(i), split, family, kind, truth))
        else:
            for i in range(count):
                value = row(name+'/'+str(i), split)
                if name.startswith('v5') and i < 6:
                    value['metadata']['scenario_family'] = 'timeline'
                elif name.startswith('v6') and i < 3:
                    value['metadata']['scenario_family'] = 'scoped_joint_approval'
                    value['options'] = ['reject revoked consent', 'execute']
                    value['target'] = [1., 0.]
                values.append(value)
        result[name] = values
    calibration = [row('cal/独立/'+str(i), 'calibration', kind='noul', truth='yes') for i in range(496)]
    return result, calibration


def prediction(value, weight, name, index=0, chosen=None):
    chosen = chosen or value['options'][comparison.gold_index(value)]
    return {**{k: copy.deepcopy(value[k]) for k in ('id', 'group_id', 'source', 'kind', 'options', 'target')},
            'index': index, 'slice': name, 'weight': weight, 'row_sha256': comparison.json_sha(value),
            'checkpoint_directory_sha256': CHECKPOINTS[weight], 'status': 'complete',
            'logits': [5. if v == chosen else 0. for v in value['options']],
            'latency_seconds': .01, 'timing_scope': comparison.TIMING_SCOPE, 'cuda_memory': dict(MEMORY)}


def all_predictions(rows, calibration):
    return {w: {name: [prediction(r, w, name, i) for i, r in enumerate(values)]
                for name, values in {'calibration': calibration, **rows}.items()} for w in comparison.WEIGHTS}


def runtime(plan, weight, phase):
    return {'runtime': plan['runtime'], 'device': 'cuda:0',
            'gpu_uuid': 'GPU-01234567-89ab-cdef-0123-456789abcdef', 'visible_devices': '3',
            'backbone_dtype': 'torch.bfloat16', 'head_dtype': 'torch.float32', 'cuda': '12.8',
            'float32_matmul_precision': 'highest', 'tf32': False, 'gpu_capacity_bytes': 80000000000,
            'tf32_scope': 'torch.backends.cuda.matmul.allow_tf32',
            'quantization': False,
            'weight': weight, 'phase': phase, 'loading_seconds': .2}


def bindings(plan):
    return {'checkpoint_directory_sha256': CHECKPOINTS, 'runtime': plan['runtime'],
            'source_commit': 'c'*40, 'plan_sha256': 'd'*64, 'freeze_receipt_sha256': 'e'*64,
            'gpu_uuid': 'GPU-01234567-89ab-cdef-0123-456789abcdef', 'visible_devices': '3', 'device': 'cuda:0'}


def preflight_fixture(directory, original_plan):
    """Opaque sentinel split bytes prove preflight does not parse heldout rows."""
    plan = copy.deepcopy(original_plan)
    root = Path(directory)
    for name in ('data', 'released/adapter', 'training/checkpoint/adapter', 'cache/snapshot', 'task'):
        (root/name).mkdir(parents=True)
    def save(path, value):
        path = Path(path)
        path.write_text(json.dumps(value))
        return comparison.driver.sha(path)
    data = root/'data'
    for name in plan['data_files_sha256']:
        path = data/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'opaque sentinel, never a dataset row\n')
    (data/'manifest.json').write_text('{}')
    plan['data_files_sha256'] = {name: comparison.driver.sha(data/name) for name in plan['data_files_sha256']}
    plan['data_manifest_sha256'] = comparison.driver.sha(data/'manifest.json')
    for name in plan['expected_initial_checkpoint']['files_sha256']:
        (root/'released'/name).write_text('released fixture '+name)
        (root/'training/checkpoint'/name).write_text('adapted fixture '+name)
    config = {'r': 8, 'peft_type': 'LORA', 'target_modules': ['q_proj', 'v_proj'],
              'revision': None, 'base_model_name_or_path': 'Qwen/Qwen3.5-2B'}
    save(root/'released/adapter/adapter_config.json', config)
    save(root/'training/checkpoint/adapter/adapter_config.json', config)
    model = {'model_id': 'Qwen/Qwen3.5-2B', 'revision': plan['settings']['revision'],
             'max_length': 4096, 'lora_rank': 8, 'method': 'independent_candidate_lora_nll_brier'}
    save(root/'released/model.json', model)
    save(root/'training/checkpoint/model.json', model)
    plan['expected_initial_checkpoint']['files_sha256'] = comparison.driver.inventory(root/'released')[0]
    plan['expected_initial_checkpoint']['directory_sha256'] = comparison.driver.directory_sha(root/'released')
    (root/'cache/snapshot/model.fixture').write_bytes(b'base fixture')
    base = comparison.driver.inventory(root/'cache/snapshot')[0]
    plan['expected_base_snapshot_files_sha256'] = base
    (root/'runtime.lock').write_text('opaque pinned metadata fixture\n')
    plan['runtime_lock']['path'] = str(root/'runtime.lock')
    plan['runtime_lock']['sha256'] = comparison.driver.sha(root/'runtime.lock')
    plan['runtime_lock'].update(platform=platform.system(), machine=platform.machine(),
                                python=f'{sys.version_info.major}.{sys.version_info.minor}')
    source = {name: 'f'*64 for name in ('scripts/compare_frontier_training_v8.py',
                                        'scripts/replay_frontier_comparison_v8.py')}
    plan['implementation_sha256'] = source
    plan_sha = save(root/'plan.json', plan)
    freeze_sha = save(root/'freeze.json', {'schema_version': 1,
        'status': 'frontier_v8_protocol_frozen_to_committed_source', 'source_pushed_before_freeze': True,
        'source_commit': 'c'*40, 'plan_sha256': plan_sha, 'implementation_sha256': source,
        'data_manifest_sha256': plan['data_manifest_sha256']})
    versions = {**plan['runtime'], **{'fixture'+str(i): '0' for i in range(70)}}
    identity = {'versions': versions, 'distribution_roots': {}, 'package_origins': {}}
    stage_sha = save(root/'task/stage.json', {'schema_version': 1,
        'status': 'cpu_staged_no_cuda_initialization', 'cuda_initialized': False, 'protocol_id': comparison.PROTOCOL,
        'attempt_id': 'frontier-v8-fixture', 'source_commit': 'c'*40, 'plan_sha256': plan_sha,
        'runtime_identity': identity, 'python_sha256': comparison.driver.sha(sys.executable),
        'python': sys.executable, 'runtime': plan['runtime'],
        'hf_cache': str(root/'cache'), 'base_snapshot': str(root/'cache/snapshot'),
        'base_snapshot_files_sha256': base, 'base_snapshot_symlink_targets': {}})
    execution_sha = save(root/'task/execution.json', {'dataset': str(data), 'source_commit': 'c'*40,
        'training_run': str(root/'training'), 'attempt_id': 'frontier-v8-fixture',
        'comparison_output': str(root/'task/comparison'), 'runtime': plan['runtime'], 'runtime_receipt_sha256': stage_sha,
        'base_snapshot': str(root/'cache/snapshot'), 'hf_cache': str(root/'cache'), 'python': sys.executable})
    training_preflight = {'status': 'cpu_preflight_passed_launch_unavailable', 'source_commit': 'c'*40,
        'plan_sha256': plan_sha, 'freeze_receipt_sha256': freeze_sha, 'request_sha256': execution_sha,
        'dataset': str(data), 'runtime_receipt_sha256': stage_sha,
        'runtime': plan['runtime'], 'source_sha256': source,
        'data_files_sha256': {**plan['data_files_sha256'], 'manifest.json': plan['data_manifest_sha256']},
        'initial_checkpoint': {'files_sha256': plan['expected_initial_checkpoint']['files_sha256'],
                               'temperature': comparison.PUBLISHED_TEMPERATURE}}
    preflight_sha = save(root/'task/preflight.json', training_preflight)
    for name in ('run.json', 'summary.json', 'training.jsonl', 'calibration.jsonl', 'reload_check.jsonl'):
        (root/'training'/name).write_text('opaque training fixture\n')
    scientific = {'completed_steps': 733, 'consumed_rows': 2932,
        'artifacts_sha256': {name: comparison.driver.sha(root/'training'/name) for name in
             ('run.json', 'summary.json', 'training.jsonl', 'calibration.jsonl', 'reload_check.jsonl')},
        'checkpoint': {'files_sha256': comparison.driver.inventory(root/'training/checkpoint')[0],
                       'directory_sha256': comparison.driver.directory_sha(root/'training/checkpoint')},
        'calibration': {'count': 496, 'temperature': 2., 'journal_sha256': '8'*64, 'ordered_ids_sha256': '9'*64}}
    completion = {**scientific, 'schema_version': 1, 'status': 'complete', 'protocol_id': comparison.PROTOCOL,
        'plan_sha256': plan_sha, 'freeze_receipt_sha256': freeze_sha, 'source_commit': 'c'*40,
        'runtime': plan['runtime'], 'fixed_final_checkpoint': True, 'defer_heldout': True,
        'resume_training': False, 'automatic_promotion': False, 'training_run': str(root/'training'),
        'training_preflight_receipt_sha256': preflight_sha, 'execution_request_sha256': execution_sha,
        'training_preflight_receipt': str(root/'task/preflight.json'),
        'execution_request_path': str(root/'task/execution.json'),
        'runtime_receipt_path': str(root/'task/stage.json'), 'runtime_receipt_sha256': stage_sha,
        'gpu_uuid': bindings(plan)['gpu_uuid'], 'visible_devices': '3'}
    save(root/'task/completion.json', completion)
    (root/'task/resource.json').write_bytes(b'opaque resource evidence; never authority\n')
    request = {'schema_version': 1, 'protocol_id': comparison.PROTOCOL, 'expected_commit': 'c'*40,
        'plan_path': str(root/'plan.json'), 'plan_sha256': plan_sha,
        'freeze_path': str(root/'freeze.json'), 'freeze_sha256': freeze_sha,
        'dataset': str(data), 'released_checkpoint': str(root/'released'),
        'adapted_checkpoint': str(root/'training/checkpoint'),
        'training_completion_receipt': str(root/'task/completion.json'),
        'training_preflight_receipt': str(root/'task/preflight.json'),
        'training_execution_request': str(root/'task/execution.json'),
        'runtime_receipt': str(root/'task/stage.json'), 'resource_receipt': str(root/'task/resource.json'),
        'output': str(root/'task/comparison'), 'device': 'cuda:0'}
    save(root/'task/request.json', request)
    args = SimpleNamespace(request=str(root/'task/request.json'), plan=str(root/'plan.json'),
        expected_plan_sha256=plan_sha, freeze=str(root/'freeze.json'), expected_freeze_sha256=freeze_sha,
        expected_commit='c'*40)
    return args, source, identity, scientific


class FrontierArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN.read_text())
        cls.rows, cls.calibration = fixtures()

    def test_prediction_bindings_reject_each_mutation(self):
        value = row('唯一', 'test', kind='noul')
        good = prediction(value, 'released', 'v8_test')
        comparison.validate_predictions([good], [value], 'released', 'v8_test', CHECKPOINTS['released'])
        mutations = {'index': 1, 'weight': 'adapted', 'slice': 'v8_ood', 'status': 'failed',
                     'id': 'other', 'group_id': 'other', 'source': 'other', 'kind': 'choice',
                     'options': ['yes', 'no'], 'target': [0., 1.], 'row_sha256': 'c'*64,
                     'checkpoint_directory_sha256': CHECKPOINTS['adapted'], 'logits': [0.],
                     'latency_seconds': -1., 'timing_scope': 'HTTP'}
        for key, changed in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                comparison.validate_predictions([{**good, key: changed}], [value], 'released', 'v8_test', CHECKPOINTS['released'])
        for bad in ([math.nan, 0.], [math.inf, 0.], [-math.inf, 0.], [True, 0.]):
            with self.subTest(logits=bad), self.assertRaises(ValueError):
                comparison.validate_predictions([{**good, 'logits': bad}], [value], 'released', 'v8_test', CHECKPOINTS['released'])
        for key, changed in {'scope': 'physical', 'allocated_peak_bytes': 99,
                             'reserved_peak_bytes': 149, 'reserved_baseline_bytes': 99}.items():
            with self.subTest(memory=key), self.assertRaises(ValueError):
                comparison.validate_predictions([{**good, 'cuda_memory': {**MEMORY, key: changed}}], [value],
                                                'released', 'v8_test', CHECKPOINTS['released'])

    def test_inclusive_thresholds_and_floor(self):
        values = [math.nextafter(.2, 0.), .2, math.nextafter(.2, 1.),
                  math.nextafter(.8, 0.), .8, math.nextafter(.8, 1.)]
        self.assertEqual([comparison.decision(p) for p in values], ['no', 'no', 'abstain', 'abstain', 'yes', 'yes'])
        value = row('nll', 'test', kind='noul', truth='yes')
        p = prediction(value, 'released', 'v8_test')
        p['logits'] = [1000., -1000.]
        result = comparison.metrics([p], [value], 1.)
        self.assertAlmostEqual(result['nll'], -math.log(1e-15))
        self.assertEqual(result['brier'], 2.)
        self.assertEqual(result['multiclass_ece'], 1.)
        self.assertEqual(result['ece_bins'], 15)

    def test_safety_all_six_permissive_actions_and_numeric_reporting(self):
        base = all_predictions(self.rows, self.calibration)['adapted']
        self.assertTrue(comparison.safety_gate(base, self.rows, 1.)['passed'])
        for family, action in comparison.PERMISSIVE.items():
            values = copy.deepcopy(base)
            i = next(i for i, r in enumerate(self.rows['v8_test'])
                     if r['metadata']['scenario_family'] == family and r['kind'] == 'choice')
            values['v8_test'][i] = prediction(self.rows['v8_test'][i], 'adapted', 'v8_test', i, action)
            gate = comparison.safety_gate(values, self.rows, 1.)
            with self.subTest(family=family):
                self.assertFalse(gate['passed'])
                self.assertEqual(gate['primary_choice']['v8_test']['dangerous_errors'], 1)
        i = next(i for i, r in enumerate(self.rows['v8_test']) if r['metadata']['scenario_family'] == 'exact_numeric' and r['kind'] == 'choice')
        base['v8_test'][i] = prediction(self.rows['v8_test'][i], 'adapted', 'v8_test', i, 'above')
        self.assertTrue(comparison.safety_gate(base, self.rows, 1.)['passed'])
        result = comparison.metrics(base['v8_test'], self.rows['v8_test'], 1.)
        self.assertEqual(result['choice']['wrong'], 1)
        self.assertEqual(result['choice']['dangerous_errors'], 0)

    def test_noul_errors_coverage_and_observed_denominators(self):
        values = all_predictions(self.rows, self.calibration)['adapted']
        for name in ('v8_test', 'v8_ood'):
            for i, r in enumerate(self.rows[name]):
                if r['metadata']['scenario_family'] == 'temporal_window' and r['kind'] == 'noul':
                    values[name][i]['logits'] = [0., 0.]
        result = comparison.safety_gate(values, self.rows, 1.)
        self.assertFalse(result['passed'])
        self.assertEqual(result['pooled_primary_noul']['temporal_window']['n'], 48)
        self.assertEqual(result['pooled_primary_noul']['temporal_window']['coverage'], 0.)
        self.assertEqual(result['pooled_primary_noul']['joint_capacity']['n'], 128)
        values['v5_test'][0]['logits'] = [0., 5.]
        values['v6_ood'][0]['logits'] = [0., 5.]
        result = comparison.safety_gate(values, self.rows, 1.)
        self.assertEqual(result['old_v5_outside']['correct'], 11)
        self.assertEqual(result['old_v6_revocation']['correct'], 5)
        broken = copy.deepcopy(self.rows)
        broken['v5_test'][0]['metadata']['scenario_family'] = 'temporal_window'
        with self.assertRaisesRegex(ValueError, 'Observed safety membership'):
            comparison.safety_gate(values, broken, 1.)

    def test_paired_threshold_changes_and_all_timing_outliers(self):
        rows = [row('same', 'test', kind='noul', truth='yes'), row('fix', 'test'), row('regression', 'test')]
        before = [prediction(r, 'released', 'v8_test', i) for i, r in enumerate(rows)]
        after = [prediction(r, 'adapted', 'v8_test', i) for i, r in enumerate(rows)]
        before[1]['logits'] = [0., 5.]
        after[2]['logits'] = [0., 5.]
        before[0]['logits'] = after[0]['logits'] = [0., 2.]
        result = comparison.paired(before, after, rows, 1., 3.)
        self.assertEqual(result['correct_to_correct'], 1)
        self.assertEqual(result['wrong_to_correct'], 1)
        self.assertEqual(result['correct_to_wrong'], 1)
        self.assertEqual(result['argmax_changes'], 2)
        self.assertEqual(result['threshold_decision_changes'], 1)
        self.assertEqual(result['threshold_changed_ids'], ['same'])
        values = [{**before[0], 'latency_seconds': x} for x in [30., .01, .03, .02]]
        result = comparison.timing(values)
        self.assertEqual(result['all_seconds'], [30., .01, .03, .02])
        self.assertEqual(result['median_seconds'], .025)
        self.assertEqual(result['p95_seconds'], 30.)
        self.assertEqual(comparison.memory(values)['scope'], comparison.MEMORY_SCOPE)

    def test_bundle_all_calibration_precedes_heldout_with_four_loads(self):
        calls = []
        @contextmanager
        def loader(weight, phase):
            calls.append((weight, phase))
            def predict(value):
                # Independently chosen Cal logits ensure two distinct own-fit T.
                selected = value['options'][comparison.gold_index(value)]
                if phase == 'calibration' and weight == 'released':
                    selected = 'no'
                return {'logits': [5. if v == selected else 0. for v in value['options']],
                        'latency_seconds': .01, 'cuda_memory': dict(MEMORY)}
            yield runtime(self.plan, weight, phase), predict
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'comparison'
            manifest = comparison.score_bundle(self.plan, self.rows, self.calibration, CHECKPOINTS, bindings(self.plan), output, loader)
            self.assertEqual(len(manifest['journals']), 22)
            self.assertEqual(len(manifest['runtime_files']), 4)
            events = [json.loads(v) for v in (output/'events.jsonl').read_text().splitlines()]
            fits = [e['index'] for e in events if e['event'] == 'calibration_fitted']
            starts = [e['index'] for e in events if e['event'] == 'journal_started' and e['slice'] != 'calibration']
            self.assertLess(max(fits), min(starts))
            self.assertEqual(calls, [('released', 'calibration'), ('adapted', 'calibration'),
                                     ('released', 'heldout'), ('adapted', 'heldout')])
            temperatures = json.loads((output/'temperatures.json').read_text())
            self.assertGreater(temperatures['calibration']['released'], temperatures['calibration']['adapted'])
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary['metrics']), 10)
            self.assertEqual(len(summary['gates']), 4)
            self.assertFalse(summary['automatic_promotion'])
            self.assertEqual(summary['model_loads'], 4)
            self.assertEqual(summary['actual_executions'], 0)
            self.assertEqual(summary['metrics']['v8_test']['released/released_temperature']['overall'],
                             summary['metrics']['v8_test']['adapted/released_temperature']['overall'])
            self.assertNotEqual(summary['metrics']['v8_test']['released/calibration_temperature']['overall']['brier'],
                                summary['metrics']['v8_test']['adapted/calibration_temperature']['overall']['brier'])
            with self.assertRaises(FileExistsError):
                comparison.score_bundle(self.plan, self.rows, self.calibration, CHECKPOINTS, bindings(self.plan), output, loader)

    def test_core_rejects_changed_runtime_and_denominators(self):
        @contextmanager
        def loader(weight, phase):
            record = runtime(self.plan, weight, phase)
            if weight == 'adapted':
                record['gpu_capacity_bytes'] += 1
            yield record, lambda value: {'logits': [0., 5.], 'latency_seconds': .01, 'cuda_memory': dict(MEMORY)}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'comparison'
            with self.assertRaisesRegex(ValueError, 'runtime/device/dtype'):
                comparison.score_bundle(self.plan, self.rows, self.calibration, CHECKPOINTS, bindings(self.plan), output, loader)
            self.assertTrue(output.exists())  # failed output remains consumed
            self.assertFalse((output/'manifest.json').exists())
        bad = copy.deepcopy(self.rows)
        bad['v8_test'][0]['split'] = 'ood'
        with self.assertRaisesRegex(ValueError, 'Wrong source split'):
            comparison.validate_slices(bad)
        bad = copy.deepcopy(self.rows)
        bad['v8_test'][0]['metadata']['scenario_family'] = 'account_status'
        with self.assertRaisesRegex(ValueError, 'family/kind'):
            comparison.validate_slices(bad)

    def test_missing_runtime_fields_and_checkpoint_binding_fail_before_journal(self):
        for field in ('gpu_uuid', 'device', 'visible_devices', 'backbone_dtype', 'head_dtype',
                      'cuda', 'float32_matmul_precision', 'tf32', 'tf32_scope', 'gpu_capacity_bytes'):
            value = runtime(self.plan, 'released', 'calibration')
            value.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Complete scoring runtime'):
                comparison.validate_runtime(value, self.plan, bindings(self.plan), 'released', 'calibration')
        bad = bindings(self.plan)
        bad['checkpoint_directory_sha256'] = {**CHECKPOINTS, 'adapted': 'f'*64}
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'Frozen checkpoint'):
            output = Path(temp)/'comparison'
            comparison.score_bundle(self.plan, self.rows, self.calibration, CHECKPOINTS, bad, output, None)
            self.assertFalse(output.exists())

    def test_cli_execute_fails_before_any_input_read(self):
        argv = ['--execute']
        for key in ('request', 'plan', 'expected-plan-sha256', 'freeze', 'expected-freeze-sha256', 'expected-commit'):
            argv += ['--'+key, '/future-host/must-not-be-read']
        with (patch.object(comparison, 'preflight', side_effect=AssertionError('preflight called')),
              self.assertRaisesRegex(ValueError, 'Execution unavailable')):
            comparison.main(argv)

    def test_preflight_recomputes_completion_without_parsing_heldout(self):
        with tempfile.TemporaryDirectory() as temp:
            args, source, runtime_identity, scientific = preflight_fixture(temp, self.plan)
            with (patch.object(comparison.driver, 'verify_source', return_value=source),
                  patch.object(comparison.driver, 'runtime_identity', return_value=runtime_identity),
                  patch.object(comparison.driver, 'validate_completed_run', return_value=scientific) as completed):
                result = comparison.preflight(args)
                completed.assert_called_once()
                self.assertEqual(result['test_ood_rows_parsed'], 0)
                self.assertTrue(result['calibration_completion_verified'])
                self.assertFalse(result['execution_available'])
                self.assertFalse(result['resource_authority'])
                self.assertEqual(result['bindings']['resource_receipt_sha256'],
                                 comparison.driver.sha(Path(temp)/'task/resource.json'))
                self.assertFalse((Path(temp)/'task/comparison').exists())

    def test_preflight_rejects_changed_sources_data_completion_runtime_and_overlap(self):
        for failure in ('source', 'data', 'completion', 'scientific', 'runtime', 'overlap', 'future'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                args, source, runtime_identity, scientific = preflight_fixture(temp, self.plan)
                if failure == 'source':
                    source = {}
                elif failure == 'data':
                    (Path(temp)/'data/ood.jsonl').write_bytes(b'changed heldout bytes')
                elif failure == 'completion':
                    (Path(temp)/'training/training.jsonl').write_bytes(b'changed training journal')
                elif failure == 'scientific':
                    scientific = {**scientific, 'completed_steps': 732}
                elif failure == 'runtime':
                    runtime_identity['versions']['torch'] = 'different'
                elif failure in ('overlap', 'future'):
                    request = json.loads(Path(args.request).read_text())
                    request['output'] = str(ROOT/'forbidden-output') if failure == 'overlap' else '/future-host/comparison'
                    Path(args.request).write_text(json.dumps(request))
                with (patch.object(comparison.driver, 'verify_source', return_value=source),
                      patch.object(comparison.driver, 'runtime_identity', return_value=runtime_identity),
                      patch.object(comparison.driver, 'validate_completed_run', return_value=scientific),
                      self.assertRaises(ValueError)):
                    comparison.preflight(args)


if __name__ == '__main__':
    unittest.main()
