"""CPU coverage diagnostics for the exact frozen v7 Train file only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from jev.data import validate_records
from scripts.audit_boundary_controls_v7 import instant_seconds, latest_credentials, latest_policy
from scripts.replay_boundary_comparison_v7 import gold, legacy_joint_state, outcome, structural_labels

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT/'data/boundary-training-v7-20261002-r1/train.jsonl'
TRAIN_SHA256 = '83fa0a04e04c43fa21a86aa799b642dad5fcb1c3e69ffb7a8b92df42c759b6c0'
SCOPE = ('Static coverage of frozen synthetic Train; no model calls, predictions, '
         'heldout row reads, new examples, training or tuning. Coverage gaps are '
         'not established model-error causes or evidence of capability gains.')


def counted(cells, names, count_name='rows'):
    return [{**dict(zip(names, key)), count_name: value} for key, value in sorted(cells.items())]


def numeric_summary(values, *, signed):
    result = {'rows': len(values), 'minimum': min(values) if values else None,
              'maximum': max(values) if values else None,
              'values': {str(key): value for key, value in sorted(Counter(values).items())}}
    if signed:
        result['relations'] = dict(sorted(Counter('below' if v < 0 else 'above' if v > 0 else 'equal'
                                                  for v in values).items()))
    return result


def summarize_rows(rows):
    if not rows or any(row.get('split') != 'train' for row in rows):
        raise ValueError('Every row must belong to Train')
    schema = validate_records(rows)
    decisions, actions, positions, conditions, axes = (Counter() for _ in range(5))
    group_sizes, groups = Counter(), defaultdict(list)
    numeric, blocked = defaultdict(list), Counter()
    structural = {name: Counter() for name in ('scope_role_kind_truth', 'outside_side_kind_truth',
                                               'negative_direction_kind_truth')}
    for row in rows:
        source, family = row['source'], row['metadata']['scenario_family']
        action, decision = outcome(row), gold(row)
        if row['target'] != [float(option == decision) for option in row['options']]:
            raise ValueError('State-derived gold differs from target: '+row['id'])
        position, kind = row['options'].index(decision), row['kind']
        condition = row['metadata'].get('condition', 'not_declared')
        decisions[(source, family, kind, decision)] += 1
        actions[(source, family, kind, action)] += 1
        positions[(source, family, kind, len(row['options']), position)] += 1
        truth_kind = kind if kind == 'choice' else kind+'/'+decision
        conditions[(source, family, condition, truth_kind)] += 1
        groups[(source, family, row['group_id'])].append(row)
        labels = structural_labels(row)
        for axis, label in labels.items():
            if axis not in ('family', 'kind', 'gold_disposition', 'condition'):
                axes[(source, family, axis, str(label))] += 1
        state = row['state']
        if family in ('timeline', 'temporal_window'):
            elapsed = instant_seconds(state['request_received_at'])-instant_seconds(state['delivered_at'])
            window = state.get('return_window_seconds', state.get('return_window_hours', 0)*3600)
            if elapsed < 0 or elapsed > window:
                numeric[(source, family, 'outside_margin_seconds')].append(elapsed if elapsed < 0 else elapsed-window)
        elif family in ('joint_capacity', 'scoped_joint_approval'):
            current = state if family == 'joint_capacity' else legacy_joint_state(state)
            latest = latest_credentials(current)
            if len(latest) == len(current['trusted_policy']['required_roles']) and all(e['status'] == 'grant' for e in latest.values()):
                numeric[(source, family, 'amount_minus_minimum_valid_capacity_cents')].append(
                    current['request']['amount_cents']-min(e['capacity_cents'] for e in latest.values()))
            else:
                blocked[(source, family, action)] += 1
        elif family in ('latest_authority', 'untrusted_policy_conflict'):
            policy = latest_policy(state)
            if policy is not None and policy['status'] == 'active':
                numeric[(source, family, 'amount_minus_active_limit_cents')].append(
                    state['request']['amount_cents']-policy['automatic_limit_cents'])
            else:
                blocked[(source, family, action)] += 1
        elif family == 'exact_numeric':
            if state['task'] == 'integer_compare':
                numeric[(source, family, 'left_minus_right_cents')].append(state['left_cents']-state['right_cents'])
            else:
                numeric[(source, family, 'ledger_entry_count')].append(len(state['ledger']))
                numeric[(source, family, 'largest_ledger_entry_cents')].append(max(e['amount_cents'] for e in state['ledger']))
        if source == 'boundary-controls-v7':
            if condition == 'scope_mismatch':
                structural['scope_role_kind_truth'][(labels['affected_scope_mismatch_fields'], labels['affected_role'], truth_kind)] += 1
            elif condition == 'equivalent_outside':
                structural['outside_side_kind_truth'][(labels['structural_axis'], truth_kind)] += 1
            elif condition == 'compare_negative':
                structural['negative_direction_kind_truth'][(action, truth_kind)] += 1
    for (source, family, _), group in groups.items():
        group_sizes[(source, family, len(group))] += 1
    expected_axes = {
        'scope_role_kind_truth': (('resource', 'operation', 'currency'), ('required_role_0', 'required_role_1')),
        'outside_side_kind_truth': (('before', 'after'),),
        'negative_direction_kind_truth': (('below', 'above'),),
    }
    crosses = {}
    for name, values in structural.items():
        expected = list(itertools.product(*expected_axes[name], ('choice', 'noul/no', 'noul/yes')))
        names = ['scope_field', 'affected_role', 'kind_truth'] if name == 'scope_role_kind_truth' else ['state_axis', 'kind_truth']
        crosses[name] = {'cells': counted(values, names),
                         'missing_cells': [dict(zip(names, key)) for key in expected if not values[key]],
                         'smallest_populated_cell': min(values.values()) if values else None,
                         'scope': 'Frozen v7 Train conditions only; counts are not a heldout coverage claim.'}
    missing = []
    for source, family in sorted({key[:2] for key in actions}):
        candidates = sorted({key[3] for key in actions if key[:2] == (source, family) and not key[3].startswith('USD ')})
        for action in candidates:
            if not actions[(source, family, 'choice', action)]:
                missing.append({'source': source, 'family': family, 'state_action': action,
                                'state_gold_rows': sum(actions[(source, family, kind, action)] for kind in ('choice', 'noul')),
                                'choice_gold_rows': 0})
    global_missing = []
    for action in sorted({key[3] for key in actions if not key[3].startswith('USD ')}):
        total = sum(n for key, n in actions.items() if key[3] == action)
        choice = sum(n for key, n in actions.items() if key[3] == action and key[2] == 'choice')
        if not choice:
            global_missing.append({'state_action': action, 'state_gold_rows': total, 'choice_gold_rows': choice})
    return {'status': 'static_train_coverage_only', 'model_calls': 0, 'heldout_row_files_read': [],
            'scope': SCOPE, 'schema': schema,
            'group_sizes': counted(group_sizes, ['source', 'family', 'rows_per_group'], 'groups'),
            'decision_gold_coverage': counted(decisions, ['source', 'family', 'kind', 'decision_gold']),
            'state_action_coverage': counted(actions, ['source', 'family', 'kind', 'state_action']),
            'gold_position_coverage': counted(positions, ['source', 'family', 'kind', 'option_count', 'gold_position_zero_based']),
            'condition_kind_truth_coverage': counted(conditions, ['source', 'family', 'condition', 'kind_truth']),
            'state_derived_axes': counted(axes, ['source', 'family', 'axis', 'value']),
            'numeric_support': [{**dict(zip(('source', 'family', 'quantity'), key)),
                                 **numeric_summary(values, signed='minus' in key[2] or key[2] == 'outside_margin_seconds')}
                                for key, values in sorted(numeric.items())],
            'authority_blocked_margin_rows': counted(blocked, ['source', 'family', 'state_action']),
            'v7_structural_cross_tabs': crosses,
            'state_actions_without_choice_gold_by_source_family': missing,
            'non_money_actions_without_any_choice_gold': global_missing,
            'limitations': ['Amounts remain exact option labels in action/decision tables; each unique amount is not a separate policy action.',
                            'Authority-blocked rows have no valid execution margin and are reported separately.',
                            'Active-policy margins include fraud routes; amount does not override fraud precedence.',
                            'Sparse cells describe authored Train support, not statistical independence or natural requests.']}


def audit_train(path):
    path = Path(path)
    if path.name != 'train.jsonl' or path.resolve().name != 'train.jsonl':
        raise ValueError('Only a file named train.jsonl is allowed; heldout filenames are excluded')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError('Train bytes differ from the pinned frozen v7 Train SHA256')
    report = summarize_rows([json.loads(line) for line in raw.splitlines()])
    report['input'] = {'file': 'train.jsonl', 'sha256': TRAIN_SHA256, 'rows': report['schema']['records']}
    report['implementation_sha256'] = {
        str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in (Path(__file__), ROOT/'jev/data.py', ROOT/'scripts/audit_boundary_controls_v7.py',
                     ROOT/'scripts/replay_boundary_comparison_v7.py')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=TRAIN)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_train(args.train)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'rows': report['input']['rows'], 'model_calls': 0}))


if __name__ == '__main__':
    main()
