"""Comparator-produced CPU bundles checked by separately authored replay.

Only manually typed states and audited visible policy constants are used. No
generator, reserved dataset row, model, CUDA call or real attempt is invoked.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import audit_frontier_controls_v8 as rules
from scripts import compare_frontier_training_v8 as comparator
from scripts import replay_frontier_comparison_v8 as independent


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {'released': 'a'*64, 'adapted': 'b'*64}
GPU = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
MEMORY = {'allocated_baseline_bytes': 100, 'reserved_baseline_bytes': 200,
          'allocated_peak_bytes': 150, 'reserved_peak_bytes': 250,
          'scope': 'model_loaded_row_forward_and_cpu_logits'}
GOLD = {'temporal_window': 'reject', 'exact_numeric': 'below',
        'joint_capacity': 'reject revoked consent', 'latest_authority': 'verify policy',
        'account_status': 'closed', 'department_route': 'manager review',
        'refund_priority': 'recall remediation'}


def typed_state(family, width):
    state = {'case_ref': 'handwritten-case', 'policy': rules.POLICIES[family]}
    if family == 'temporal_window':
        state.update(recorded_delivery_at='2026-01-01T00:00:00Z', correction_authority='signed-owner',
            delivery_corrections=[{'verified_signature': True, 'issuer_role': 'signed-owner',
                'case_ref': 'handwritten-case', 'sequence': i, 'seconds': 1, 'direction': 'delay'} for i in range(width)],
            return_window_seconds=10, request_received_at='2026-01-01T00:00:00Z', exception_approved=False)
    elif family == 'exact_numeric':
        def ledger(side, credit, debit):
            return [{'entry_ref': side+'/'+direction+'/'+str(i), 'amount_cents': amount, 'direction': direction}
                    for direction, amount in (('credit', credit), ('debit', debit)) for i in range(width)]
        state.update(task='integer_compare', currency='USD',
                     left_ledger=ledger('left', 2, 3), right_ledger=ledger('right', 4, 1))
    elif family == 'joint_capacity':
        scope = {'resource': 'invoice', 'operation': 'pay', 'currency': 'USD'}
        roles = ['role'+str(i) for i in range(width+1)]
        state.update(request={**scope, 'amount_cents': 5}, trusted_policy={'required_roles': roles},
            signed_events=[{'issuer_role': role, 'sequence': 1, 'verified_signature': True,
                            'credential_scope': dict(scope), 'status': 'revoke' if i == width else 'grant',
                            'capacity_cents': 10} for i, role in enumerate(roles)])
    elif family == 'latest_authority':
        topics = ['topic'+str(i) for i in range(width+1)]
        state.update(request={'department': 'finance', 'amount_cents': 5, 'confirmed_fraud': False},
            required_topics=topics, policy_authorities={topic: 'authority'+str(i) for i, topic in enumerate(topics)},
            signed_policy_registry=[{'topic': topic, 'department': 'finance',
                'credential_scope': {'department': 'finance', 'topic': topic},
                'issuer_role': 'authority'+str(i), 'verified_signature': True, 'revision': 1,
                'automatic_limit_cents': 10, 'status': 'withdrawn' if i == width else 'active'}
                for i, topic in enumerate(topics)])
    elif family == 'account_status':
        accounts = ['account'+str(i) for i in range(width)]
        state.update(required_accounts=accounts, events=[{'account': account, 'sequence': 1,
                     'status': 'closed' if i == width-1 else 'active'} for i, account in enumerate(accounts)])
    elif family == 'department_route':
        departments = ['department'+str(i) for i in range(width)]
        state.update(policy_authority='owner', request_lines=[{'department': department,
            'amount_cents': 11, 'confirmed_fraud': False} for department in departments],
            signed_policies=[{'department': department, 'credential_scope': department,
                'issuer_role': 'owner', 'verified_signature': True, 'revision': 1,
                'status': 'active', 'approval_limit_cents': 10} for department in departments])
    elif family == 'refund_priority':
        state.update(purchase_document_verified=False, trusted_policy={'currency': 'USD', 'automatic_limit_cents': 10},
            items=[{'item_ref': 'item'+str(i), 'amount_cents': 2, 'verified_recall': i == 0,
                    'defect_confirmed': False, 'seal_intact': True} for i in range(width)])
    else:
        raise AssertionError('Unreviewed hand fixture family')
    return state


def primary_row(identity, split, family, kind, ordinal):
    width = {'calibration': 3, 'test': 5, 'ood': 6}[split]
    choices = list(rules.OPTIONS.get(family, ('below', 'equal', 'above')))
    truth = GOLD[family]
    proposed = truth if ordinal % 2 else next(value for value in choices if value != truth)
    options = choices if kind == 'choice' else ['no', 'yes']
    expected = truth if kind == 'choice' else ('yes' if proposed == truth else 'no')
    return {'id': identity, 'group_id': identity+'/parent', 'source': 'frontier-controls-v8',
            'split': split, 'state': typed_state(family, width), 'kind': kind,
            'question': rules.CHOICE_QUESTIONS[0] if kind == 'choice' else rules.NOUL_TEMPLATE.format(proposed_outcome=proposed),
            'options': options, 'target': [float(value == expected) for value in options],
            'metadata': {'scenario_family': family, 'condition': 'hand-authored-contract',
                         'composition_width': width, 'layout': 'forward'}}


def legacy_row(identity, split, origin, ordinal):
    family, options, expected = 'state_tracking', ['active', 'paused', 'closed', 'unknown'], 'closed'
    state = {'account': 'a', 'events': [{'account': 'a', 'sequence': 1, 'status': 'closed'}]}
    if origin == 'v5' and ordinal < 6:
        family, options, expected = 'timeline', ['accept', 'reject', 'review'], 'reject'
        state = {'delivered_at': '2026-01-01T00:00:00Z', 'request_received_at': '2026-01-03T00:00:00Z',
                 'return_window_hours': 24, 'exception_approved': False}
    elif origin == 'v6' and ordinal < 3:
        family, options, expected = 'scoped_joint_approval', list(rules.OPTIONS['joint_capacity']), 'reject revoked consent'
        scope = {'resource': 'invoice', 'operation': 'pay', 'currency': 'USD'}
        state = {'request': {**scope, 'amount_cents': 5}, 'trusted_policy': {'required_roles': ['owner', 'reviewer']},
                 'signed_events': [{**scope, 'issuer_role': role, 'sequence': 1, 'verified_signature': True,
                                    'credential_scope': dict(scope), 'status': 'revoke' if role == 'owner' else 'grant',
                                    'capacity_cents': 10} for role in ('owner', 'reviewer')]}
    return {'id': identity, 'group_id': identity+'/parent', 'source': 'hand-authored-legacy-fixture',
            'split': split, 'state': state, 'kind': 'choice', 'question': 'Which supplied outcome follows?',
            'options': options, 'target': [float(value == expected) for value in options],
            'metadata': {'scenario_family': family, 'condition': 'hand-authored-contract'}}


def data_fixture():
    rows = {}
    for name, count in comparator.COUNTS.items():
        split = name.rsplit('_', 1)[1]
        if name.startswith('v8'):
            rows[name] = [primary_row(name+'/'+family+'/'+kind+'/'+str(i), split, family, kind, i)
                for family, family_count in comparator.FAMILY_ROWS.items() for kind in ('choice', 'noul')
                for i in range(family_count//2)]
        else:
            rows[name] = [legacy_row(name+'/'+str(i), split, name[:2], i) for i in range(count)]
    calibration = [primary_row('cal/独立/'+str(i), 'calibration', 'temporal_window', 'noul', i) for i in range(496)]
    return rows, calibration


def runtime(plan, weight, phase):
    return {'runtime': plan['runtime'], **plan['runtime_settings'], 'device': 'cuda:0', 'gpu_uuid': GPU,
            'visible_devices': '3', 'gpu_capacity_bytes': 80000000000,
            'weight': weight, 'phase': phase, 'loading_seconds': .2}


class ComparatorReplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT/'reports/frontier-v8-training-protocol-20261003/training-plan.json').read_text())
        cls.rows, cls.calibration = data_fixture()
        cls.bindings = {'checkpoint_directory_sha256': CHECKPOINTS, 'runtime': cls.plan['runtime'],
            'source_commit': 'c'*40, 'plan_sha256': 'd'*64, 'freeze_receipt_sha256': 'e'*64,
            'gpu_uuid': GPU, 'visible_devices': '3', 'device': 'cuda:0'}
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)/'comparison'
        @contextmanager
        def loader(weight, phase):
            def predict(row):
                correct = row['target'].index(1.)
                logits = [5. if i == correct else 0. for i in range(len(row['options']))]
                if weight == 'adapted' and row['id'].endswith('/0'):
                    logits = list(reversed(logits))
                return {'logits': logits, 'latency_seconds': 30. if row['id'].endswith('/0') else .01,
                        'cuda_memory': dict(MEMORY)}
            yield runtime(cls.plan, weight, phase), predict
        comparator.score_bundle(cls.plan, cls.rows, cls.calibration, CHECKPOINTS, cls.bindings, cls.output, loader)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_independent_replay_matches_all_metrics_gates_pairs_and_efficiency(self):
        proof = independent.validate_bundle(self.output, self.plan, self.rows, self.calibration, self.bindings)
        self.assertEqual(proof['journal_count'], 22)
        self.assertEqual(proof['runtime_count'], 4)
        self.assertEqual(proof['predictions_per_weight'], 2392)
        self.assertEqual(len(proof['gates']), 4)
        self.assertTrue(proof['gates']['released/released_temperature']['passed'])
        self.assertFalse(proof['gates']['adapted/calibration_temperature']['passed'])

    def test_replay_rejects_rehashed_events_runtime_and_scientific_summary(self):
        for name in ('events', 'runtime', 'summary'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)/'comparison'
                shutil.copytree(self.output, output)
                manifest = json.loads((output/'manifest.json').read_text())
                if name == 'events':
                    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
                    events[4], events[6] = events[6], events[4]
                    for i, event in enumerate(events):
                        event['index'] = i
                    (output/'events.jsonl').write_text(''.join(json.dumps(value)+'\n' for value in events))
                    manifest['events_sha256'] = independent.sha(output/'events.jsonl')
                elif name == 'runtime':
                    path = output/'adapted_heldout.runtime.json'
                    value = json.loads(path.read_text())
                    value['tf32'] = True
                    path.write_text(json.dumps(value))
                    manifest['runtime_files'][path.name] = independent.sha(path)
                else:
                    path = output/'summary.json'
                    value = json.loads(path.read_text())
                    value['metrics']['v8_test']['adapted/calibration_temperature']['overall']['accuracy'] = 1.
                    path.write_text(json.dumps(value))
                    manifest['summary_sha256'] = independent.sha(path)
                (output/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    independent.validate_bundle(output, self.plan, self.rows, self.calibration, self.bindings)

    def test_replay_rejects_unbound_journal_sha_and_prediction_order(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'comparison'
            shutil.copytree(self.output, output)
            path = output/'released_v8_test.jsonl'
            records = path.read_text().splitlines()
            records[0], records[1] = records[1], records[0]
            path.write_text('\n'.join(records)+'\n')
            with self.assertRaises(ValueError):
                independent.validate_bundle(output, self.plan, self.rows, self.calibration, self.bindings)

    def test_exclusive_replay_receipt_with_explicitly_mocked_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('data', 'released', 'training/checkpoint'):
                (root/name).mkdir(parents=True)
            completion = root/'completion.json'
            completion.write_text(json.dumps({'training_run': str(root/'training')}))
            request = {name: str(root/name) for name in ('dataset', 'released_checkpoint', 'adapted_checkpoint',
                'training_preflight_receipt', 'training_execution_request', 'runtime_receipt', 'resource_receipt')}
            request.update(dataset=str(root/'data'), released_checkpoint=str(root/'released'),
                           adapted_checkpoint=str(root/'training/checkpoint'), training_completion_receipt=str(completion))
            (root/'request.json').write_text(json.dumps(request))
            for name in ('plan.json', 'freeze.json'):
                (root/name).write_text('{}')
            args = SimpleNamespace(output=str(root/'receipt.json'), comparison=str(self.output),
                request=str(root/'request.json'), plan=str(root/'plan.json'), freeze=str(root/'freeze.json'),
                expected_plan_sha256='d'*64, expected_freeze_sha256='e'*64, expected_commit='c'*40)
            # Only provenance is mocked; journal hashes, state gold, arithmetic,
            # event sequence, runtime records, receipt and exclusive I/O are real.
            with patch.object(independent, 'provenance', return_value=(self.plan, self.rows, self.calibration, self.bindings)) as provenance:
                result = independent.replay(args)
                self.assertEqual(provenance.call_count, 2)
                self.assertEqual(result['status'], 'independent_replay_passed')
                self.assertEqual(result['model_calls'], 0)
                self.assertEqual(result['GPU_actions'], 0)
                self.assertEqual(result['resource_controller_validation'], 'not_performed')
                self.assertEqual(result['restoration_validation'], 'not_performed')
                self.assertTrue((root/'receipt.json.consumed.json').is_file())
                with self.assertRaises(ValueError):
                    independent.replay(args)
                (root/'receipt.json').unlink()
                with self.assertRaises(FileExistsError):
                    independent.replay(args)


if __name__ == '__main__':
    unittest.main()
