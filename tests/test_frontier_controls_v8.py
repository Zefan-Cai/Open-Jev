"""Provisional CPU generation, supervision and essential-composition checks."""
from collections import Counter, defaultdict
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev.data import validate_records
from jev import frontier_controls_v8 as data


class FrontierV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(data.records())

    def test_deterministic_counts_and_schema(self):
        self.assertEqual(self.rows, list(data.records()))
        summary = validate_records(self.rows)
        self.assertEqual(summary['records'], 2976)
        self.assertEqual(summary['groups'], 42)
        self.assertEqual(summary['splits'], dict(calibration=496, ood=496, test=496, train=992, validation=496))
        self.assertEqual(summary['kinds'], dict(choice=1488, noul=1488))

    def test_four_presentations_no_truth_layout_shortcut(self):
        groups = defaultdict(list)
        for row in self.rows:
            groups[(row['group_id'], row['metadata']['case_index'])].append(row)
        for rows in groups.values():
            self.assertEqual(len(rows), 4)
            choice = [r for r in rows if r['kind'] == 'choice']
            noul = [r for r in rows if r['kind'] == 'noul']
            self.assertEqual({r['metadata']['layout'] for r in choice}, {'forward', 'reverse'})
            self.assertEqual(len({r['metadata']['layout'] for r in noul}), 1)
            self.assertEqual(Counter(r['target'][1] for r in noul), {0.: 1, 1.: 1})
            for row in rows:
                result = data.oracle(row['metadata']['scenario_family'], row['state'])
                result = result if row['kind'] == 'choice' else 'yes' if result == row['metadata']['proposed_outcome'] else 'no'
                self.assertEqual(row['options'][row['target'].index(1.)], result)

    def test_declared_disposition_schedule_is_equal_for_every_scaffold(self):
        schedules = {
            'temporal_window': ['accept']*4+['reject']*4+['review']*4,
            'joint_capacity': ['execute']*8+['request missing consent']*8+['reject revoked consent']*8+['request higher capacity']*8,
            'latest_authority': ['automatic processing']*8+['capacity review']*8+['fraud review']*8+['verify policy']*8,
            'account_status': ['active']*3+['paused']*3+['closed']*3+['unknown']*3,
            'department_route': ['approve']*4+['manager review']*4+['security review']*4,
            'refund_priority': ['automatic reimbursement']*3+['recall remediation']*3+['reject request']*3+['review reimbursement']*3,
        }
        for row in self.rows:
            if row['metadata']['presentation_index'] == 0 and row['metadata']['scenario_family'] in schedules:
                expected = schedules[row['metadata']['scenario_family']][row['metadata']['case_index']]
                self.assertEqual(row['options'][row['target'].index(1.)], expected)

    def test_every_essential_component_can_change_an_outcome(self):
        rows = {(r['metadata']['scenario_family'], r['metadata']['composition_width'], r['metadata']['case_index']): r
                for r in self.rows if r['metadata']['presentation_index'] == 0}
        for width in range(1, 7):
            temporal = [rows[('temporal_window', width, index)] for index in (0, 3)]
            for index in range(width):
                changed = []
                for row in temporal:
                    state = copy.deepcopy(row['state']); state['delivery_corrections'].pop(index)
                    changed.append(data.oracle('temporal_window', state) != 'accept')
                self.assertTrue(any(changed))
            for family, index, key in (('joint_capacity', 4, 'signed_events'), ('latest_authority', 4, 'signed_policy_registry'),
                                       ('account_status', 0, 'events')):
                row = rows[(family, width, index)]
                for position in range(len(row['state'][key])):
                    state = copy.deepcopy(row['state']); state[key].pop(position)
                    self.assertNotEqual(data.oracle(family, state), data.oracle(family, row['state']))
            row = rows[('department_route', width, 2)]
            for position in range(width):
                state = copy.deepcopy(row['state'])
                state['request_lines'][position]['amount_cents'] = state['signed_policies'][position]['approval_limit_cents']+1
                self.assertEqual(data.oracle('department_route', state), 'manager review')
            row = rows[('refund_priority', width, 2)]
            for position in range(width):
                state = copy.deepcopy(row['state']); state['items'][position].update(defect_confirmed=False, seal_intact=False)
                self.assertEqual(data.oracle('refund_priority', state), 'reject request')
            row = rows[('exact_numeric', width, 4)]
            for position in range(len(row['state']['ledger'])):
                state = copy.deepcopy(row['state']); state['ledger'].pop(position)
                self.assertNotEqual(data.oracle('exact_numeric', state), data.oracle('exact_numeric', row['state']))

    def test_no_neutral_ledger_padding_and_positions_balanced(self):
        positions = defaultdict(Counter); ranks = defaultdict(Counter)
        for row in self.rows:
            state = row['state']; family = row['metadata']['scenario_family']
            if row['kind'] == 'choice':
                positions[(row['split'], family, state.get('task'), len(row['options']))][row['target'].index(1.)] += 1
            if family == 'exact_numeric':
                for key in ('ledger', 'left_ledger', 'right_ledger'):
                    if key in state and row['metadata']['composition_width'] > 1:
                        credit = {e['amount_cents'] for e in state[key] if e['direction'] == 'credit'}
                        debit = {e['amount_cents'] for e in state[key] if e['direction'] == 'debit'}
                        self.assertFalse(credit.intersection(debit))
                if state['task'] == 'balance' and row['metadata']['presentation_index'] == 0:
                    ranks[row['split']][row['metadata']['money_gold_numeric_rank']] += 1
                    ranks[(row['split'], row['metadata']['composition_width'])][row['metadata']['money_gold_numeric_rank']] += 1
        for key, counts in positions.items():
            self.assertEqual(set(counts), set(range(key[-1])))
            self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
        for counts in ranks.values():
            self.assertEqual(set(counts), set(range(5)))
            self.assertLessEqual(max(counts.values())-min(counts.values()), 1)

    def test_generation_reads_no_data_or_models_and_opaque_ids(self):
        with (patch.object(Path, 'read_text', side_effect=AssertionError('File read forbidden')),
              patch.object(Path, 'read_bytes', side_effect=AssertionError('File read forbidden'))):
            rows = list(data.records())
        forbidden = ('/train/', '/test/', '/ood/', 'at_start', 'revoke_first', 'balance_negative')

        def strings(value):
            if isinstance(value, dict):
                return [s for k, v in value.items() for s in strings(k)+strings(v)]
            if isinstance(value, list):
                return [s for v in value for s in strings(v)]
            return [value] if isinstance(value, str) else []

        for row in rows:
            for value in strings(row['state']):
                if value.startswith(('case-', 'entry-', 'item-', 'authority-', 'account-', 'topic-', 'resource-', 'department-')):
                    self.assertRegex(value, r'^[a-z]+-[0-9a-f]{20}$')
                    self.assertFalse(any(word in value for word in forbidden))

    def test_existing_output_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); sentinel = output/'sentinel'; sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                data.build(output)
            self.assertEqual(sentinel.read_text(), 'keep')
            self.assertEqual(list(output.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main()
