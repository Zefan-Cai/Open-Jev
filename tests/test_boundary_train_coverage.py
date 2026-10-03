"""Hand-authored coverage fixtures and Train-only file-access checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_boundary_train_coverage as audit


def fixture(state, family, action, *, kind='choice', source='boundary-controls-v7', identity='row'):
    choices = {
        'joint_capacity': ['execute', 'request missing consent', 'reject revoked consent', 'request higher capacity'],
        'latest_authority': ['automatic processing', 'capacity review', 'fraud review', 'verify policy'],
        'state_tracking': ['active', 'paused', 'closed', 'unknown'],
    }
    options = choices[family] if kind == 'choice' else ['no', 'yes']
    target = [float(label == action) for label in options] if kind == 'choice' else [0., 1.]
    question = ('Apply the supplied exact rule to the recorded facts. Which stated outcome follows?' if kind == 'choice'
                else f"Does the supplied policy establish the outcome '{action}'? Use the explicit facts.")
    return {'id': identity, 'group_id': 'group/'+identity, 'source': source, 'split': 'train',
            'state': state, 'question': question,
            'kind': kind, 'options': options, 'target': target,
            'metadata': {'scenario_family': family, 'family': 'policy', 'condition': 'equality',
                         'template_id': 'fixture', 'entity_ids': [],
                         'proposed_outcome': action if kind == 'noul' else None,
                         'provenance': {'type': 'synthetic', 'license': 'CC0-1.0', 'split_policy': 'Train-only fixture',
                                        'generator_version': 'fixture', 'seed': 0, 'group_index': 0, 'variant': 0}}}


def joint(amount):
    request = {'resource': 'vault', 'operation': 'release', 'currency': 'USD', 'amount_cents': amount}
    scope = {key: request[key] for key in ('resource', 'operation', 'currency')}
    return {'request': request, 'trusted_policy': {'required_roles': ['A', 'B'], 'rules': 'fixture'},
            'signed_events': [{'issuer_role': role, 'verified_signature': True, 'credential_scope': copy.deepcopy(scope),
                               'sequence': 1, 'status': 'grant', 'capacity_cents': capacity}
                              for role, capacity in (('A', 100), ('B', 200))]}


def authority(amount, *, withdrawn=False):
    return {'request': {'department': 'care', 'amount_cents': amount, 'confirmed_fraud': False},
            'policy_authority': 'board', 'trust_contract': 'fixture',
            'signed_policy_registry': [{'department': 'care', 'credential_scope': 'care', 'issuer_role': 'board',
                                       'verified_signature': True, 'revision': 2,
                                       'status': 'withdrawn' if withdrawn else 'active', 'automatic_limit_cents': 100}]}


class TrainCoverageTests(unittest.TestCase):
    def test_unsafe_heldout_filename_rejected_before_read(self):
        for name in ('test.jsonl', 'ood.jsonl', 'calibration.jsonl', 'validation.jsonl'):
            with self.subTest(name=name), patch.object(Path, 'read_bytes', side_effect=AssertionError('read forbidden')):
                with self.assertRaisesRegex(ValueError, 'Only a file named train.jsonl'):
                    audit.audit_train(Path('/not-read')/name)

    def test_symlink_to_heldout_filename_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.jsonl'
            path.symlink_to(Path(directory)/'test.jsonl')
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('read forbidden')):
                with self.assertRaisesRegex(ValueError, 'Only a file named train.jsonl'):
                    audit.audit_train(path)

    def test_wrong_hash_rejected_before_json_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.jsonl'; path.write_bytes(b'not json\n')
            with self.assertRaisesRegex(ValueError, 'pinned frozen v7 Train SHA256'):
                audit.audit_train(path)

    def test_non_train_rows_rejected(self):
        row = fixture(joint(100), 'joint_capacity', 'execute')
        for split in ('test', 'ood', 'calibration', 'validation'):
            row['split'] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'Every row must belong to Train'):
                audit.summarize_rows([row])

    def test_exact_state_margins_and_authority_blocked_exclusion(self):
        rows = []
        for amount in (99, 100, 101):
            rows.append(fixture(joint(amount), 'joint_capacity', 'request higher capacity' if amount > 100 else 'execute',
                                identity='joint/'+str(amount)))
            rows.append(fixture(authority(amount), 'latest_authority', 'capacity review' if amount > 100 else 'automatic processing',
                                identity='policy/'+str(amount)))
        rows.append(fixture(authority(100000, withdrawn=True), 'latest_authority', 'verify policy', identity='withdrawn'))
        report = audit.summarize_rows(rows)
        support = {row['quantity']: row for row in report['numeric_support']}
        for quantity in ('amount_minus_minimum_valid_capacity_cents', 'amount_minus_active_limit_cents'):
            self.assertEqual(support[quantity]['values'], {'-1': 1, '0': 1, '1': 1})
            self.assertEqual(support[quantity]['relations'], {'above': 1, 'below': 1, 'equal': 1})
        self.assertEqual(report['authority_blocked_margin_rows'], [
            {'source': 'boundary-controls-v7', 'family': 'latest_authority', 'state_action': 'verify policy', 'rows': 1}])

    def test_missing_choice_action_and_no_input_mutation(self):
        row = fixture({'account': 'A', 'events': []}, 'state_tracking', 'unknown', kind='noul',
                      source='frontier-controls-v4')
        before = copy.deepcopy(row)
        report = audit.summarize_rows([row])
        self.assertEqual(report['non_money_actions_without_any_choice_gold'], [
            {'state_action': 'unknown', 'state_gold_rows': 1, 'choice_gold_rows': 0}])
        self.assertEqual(row, before)
        row['target'] = [1., 0.]
        with self.assertRaisesRegex(ValueError, 'State-derived gold differs'):
            audit.summarize_rows([row])

    @unittest.skipUnless(audit.TRAIN.is_file(), 'Frozen Train artifact is not part of a clean checkout')
    def test_exact_frozen_train_and_only_train_row_file_read(self):
        original = Path.read_bytes; reads = []

        def observed(path):
            reads.append(path)
            return original(path)

        with patch.object(Path, 'read_bytes', observed):
            report = audit.audit_train(audit.TRAIN)
        self.assertEqual([path.name for path in reads if path.suffix == '.jsonl'], ['train.jsonl'])
        self.assertEqual(report['input']['rows'], 3792)
        self.assertEqual(report['schema']['kinds'], {'choice': 2588, 'noul': 1204})
        self.assertEqual(report['non_money_actions_without_any_choice_gold'], [
            {'state_action': 'review reimbursement', 'state_gold_rows': 20, 'choice_gold_rows': 0},
            {'state_action': 'security review', 'state_gold_rows': 100, 'choice_gold_rows': 0},
            {'state_action': 'unknown', 'state_gold_rows': 100, 'choice_gold_rows': 0}])
        cross = report['v7_structural_cross_tabs']['scope_role_kind_truth']
        self.assertEqual(len(cross['cells']), 18)
        self.assertEqual(cross['missing_cells'], [])
        self.assertEqual(cross['smallest_populated_cell'], 1)
        support = [row for row in report['numeric_support'] if row['family'] == 'joint_capacity']
        self.assertEqual(support[0]['values'], {'0': 64, '1': 64})


if __name__ == '__main__':
    unittest.main()
