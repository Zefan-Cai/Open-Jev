"""Independent input-only full-mixture gate; never materialize or run a model.

Exact published inputs and ordered whole-parent membership are checked before
five-mode isolation. Old heldout bodies and labels never enter the receipt.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

from jev.data import SPLITS, validate_records
from scripts.audit_frontier_isolation_v8 import (PROSE_KEYS, canonical,
    collision_index, digest, norm, normalize_state, row_fingerprints)


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'reports/boundary-controls-v7-20261002/prepared-training/manifest.json':
        'adcf3c66af2c6398c9039220172d10ce3398c455f40d7fc9bfc0af639a33aae0',
    'reports/boundary-controls-v7-20261002/prepared-training/comparison-plan.json':
        'bab40284419d8ff0bfcdca818047cc96581d0d4e69a7fb891f1c08fe0ec49f35',
    'reports/frontier-v8-data-20261003/freeze-receipt.json':
        'cd554fe00bd592febdf01a0b41c20de676054e54484a1bd27dc8f86ee4e20232',
    'reports/frontier-v8-data-20261003/contract.json':
        '30daab0f14711bc8e707e954fb38ba4f4ea73ec394589e0386c2e7e1d0b664ca',
    'scripts/audit_frontier_isolation_v8.py':
        '5d06384058a1ee13508c8ea1f98ce7f4279782a7ab49f0e8594c266911468a81',
}
VERSIONS = {'v4': 'frontier-controls-v4', 'v5': 'temporal-windows-v5',
    'v6': 'original-policy-controls-v6-candidate', 'v7': 'boundary-controls-v7'}
OBSERVED = {'v4': ROOT/'data/frontier-controls-v4-20261002',
    'v5': ROOT/'data/temporal-windows-v5-20261002-r1',
    'v6': ROOT/'runs/openjev-hf-data-20261002/original-policy-v6-candidate',
    'v7': ROOT/'data/boundary-controls-v7-20261002-r1'}
LOCKS = {'v4': ('reports/frontier-controls-v4/continued-2b-20261002/run.lock.json', 'selection'),
    'v5': ('reports/temporal-windows-v5-20261002/ms-2b-fixed-pilot/pilot/pilot.lock.json', 'v5_selection')}
BLOCKING_MODES = ('compiled_visible', 'wording_order_identity',
                  'normalized_visible_body', 'whole_parent_scaffold')
SOURCE_FILES = ('scripts/prepare_frontier_training_v8.py', 'scripts/audit_frontier_training_v8.py',
                'jev/train.py', 'jev/data.py', 'jev/api.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def legacy_json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def checked_bytes(path, checksum):
    raw = Path(path).read_bytes()
    require(sha(raw) == checksum, 'Pinned input bytes changed: ' + str(path))
    return raw


def checked_rows(path, checksum, count, split, source=None):
    raw = checked_bytes(path, checksum)
    require(raw.endswith(b'\n') and all(raw.splitlines()), 'Complete nonblank JSONL lines required')
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == count and all(r['split'] == split for r in rows), 'Input count/split differs')
    if source is not None:
        require(all(r['source'] == source for r in rows), 'Source provenance version differs')
    return raw, rows


def safe_schema(rows):
    try:
        return validate_records(rows)
    except ValueError:
        raise ValueError('Input schema/provenance/parent validation failed') from None


def indexed(rows):
    """Retain references privately for counts; exported identifiers are hashes."""
    index = collision_index(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    return index, groups


def compare_indices(left, right):
    a, groups = left
    b, other_groups = right
    modes, blocking_groups = {}, set()
    for mode in (*BLOCKING_MODES, 'magnitude_time_scaffold'):
        shared = sorted(set(a[mode]) & set(b[mode]))
        touched = {g for fp in shared for _, g in a[mode][fp]}
        other = {g for fp in shared for _, g in b[mode][fp]}
        if mode in BLOCKING_MODES:
            blocking_groups.update(touched)
        modes[mode] = {'matched_fingerprints': len(shared),
            'fingerprint_set_sha256': digest(shared),
            'left_whole_parents': len(touched), 'right_whole_parents': len(other),
            'left_parent_closure_rows': sum(len(groups[g]) for g in touched),
            'right_parent_closure_rows': sum(len(other_groups[g]) for g in other),
            'blocking': mode in BLOCKING_MODES}
    return modes, blocking_groups


def legacy_account_core(state):
    """Preserve actual subject-to-event edges independently of array positions."""
    require(set(state) == {'account', 'events', 'rule'}, 'Not the reviewed legacy account schema')
    require(isinstance(state['account'], str) and isinstance(state['rule'], str), 'Typed account rule required')
    accounts = defaultdict(list)
    for event in state['events']:
        require(set(event) == {'account', 'sequence', 'status'} and isinstance(event['account'], str)
            and type(event['sequence']) is int and isinstance(event['status'], str), 'Typed legacy event required')
        accounts[event['account']].append((event['sequence'], norm(event['status'])))
    return {'rule': norm(state['rule']),
        'target_account_events': sorted(accounts.pop(state['account'], [])),
        'other_account_event_bundles': sorted([sorted(v) for v in accounts.values()], key=canonical)}


def legacy_account_query(row):
    """Only the published v4 question forms establish the same semantic query."""
    base = "What is this account's current status?"
    context = 'Resolve the case from the supplied facts only. '
    question = row['question']
    if row['kind'] == 'choice':
        require(question in (base, context+base), 'Unreviewed legacy account question')
        return {'query': 'current_status_of_specified_account', 'kind': 'choice'}
    match = re.fullmatch(r"Does the supplied policy establish the outcome '(active|paused|closed|unknown)'\? "
        + '(?:'+re.escape(context)+')?'+re.escape(base), question)
    require(row['kind'] == 'noul' and match is not None, 'Unreviewed legacy account predicate question')
    return {'query': 'current_status_of_specified_account', 'kind': 'noul', 'proposition': match[1]}


def rule_fields(value, path=()):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PROSE_KEYS | {'rule'} and isinstance(item, str):
                result['/'.join(path + (key,))] = norm(item)
            else:
                result.update(rule_fields(item, path + (key,)))
    elif isinstance(value, list):
        # Strict comparison is conservative when rule prose exists inside records.
        for i, item in enumerate(value):
            result.update(rule_fields(item, path + (str(i),)))
    return result


def confirm_clones(train, observed):
    left, right = defaultdict(list), defaultdict(list)
    for rows, index in ((train, left), (observed, right)):
        for row in rows:
            index[row_fingerprints(row)['normalized_visible_body']].append(row)
    evidence, confirmed_groups = [], set()
    for fp in sorted(set(left) & set(right)):
        a, b = left[fp], right[fp]
        normalized = {digest(normalize_state(r['state'])) for r in a + b}
        rules = {digest(rule_fields(r['state'])) for r in a + b}
        try:
            cores = {digest(legacy_account_core(r['state'])) for r in a + b}
            queries = {digest(legacy_account_query(r)) for r in a + b}
            query_equal = len(queries) == 1
            confirmed = len(cores) == 1 and len(normalized) == 1 and len(rules) == 1 and query_equal
        except ValueError:
            confirmed = query_equal = False
        if confirmed:
            confirmed_groups.update(r['group_id'] for r in a)
        evidence.append({'fingerprint': fp, 'all_normalized_typed_states_equal': len(normalized) == 1,
            'all_rule_fields_equal': len(rules) == 1,
            'all_reviewed_visible_queries_equal': query_equal,
            'legacy_subject_and_non_subject_reference_graph_equal': confirmed,
            'train_matching_rows': len(a), 'observed_matching_rows': len(b),
            'representative_train_id_sha256': digest(a[0]['id']),
            'representative_observed_id_sha256': digest(b[0]['id']),
            'state_field_categories': sorted(a[0]['state']),
            'source': a[0]['source'], 'family': a[0]['metadata'].get('scenario_family', 'unknown')})
    return {'fingerprints': len(evidence), 'confirmed_typed_account_clone_fingerprints':
        sum(e['legacy_subject_and_non_subject_reference_graph_equal'] for e in evidence),
        'matching_train_rows': sum(e['train_matching_rows'] for e in evidence),
        'matching_observed_rows': sum(e['observed_matching_rows'] for e in evidence),
        'evidence': evidence}, confirmed_groups


def diagnose(train, new_reserved, observed):
    current, reserved, old = indexed(train), indexed(new_reserved), indexed(observed)
    primary, primary_groups = compare_indices(current, reserved)
    exclusions, observed_groups = compare_indices(current, old)
    primary_groups.update(set(current[1]) & set(reserved[1]))
    observed_groups.update(set(current[1]) & set(old[1]))
    identifier_overlap = {
        'train_vs_new_reserved_row_ids': len({r['id'] for r in train} & {r['id'] for r in new_reserved}),
        'train_vs_observed_row_ids': len({r['id'] for r in train} & {r['id'] for r in observed})}
    cross = {mode: sum(len({s for s, _ in refs}) > 1 for refs in reserved[0][mode].values())
             for mode in BLOCKING_MODES}
    clones, proven_groups = confirm_clones(train, observed)
    blocked = bool(primary_groups or observed_groups or any(cross.values()) or any(identifier_overlap.values()))
    whole_families = Counter()
    for fp in set(current[0]['whole_parent_scaffold']) & set(old[0]['whole_parent_scaffold']):
        for source, family in {(current[1][g][0]['source'], current[1][g][0]['metadata'].get('scenario_family', 'unknown'))
                               for _, g in current[0]['whole_parent_scaffold'][fp]}:
            whole_families[source+'|'+family] += 1
    return {'status': 'independent_frontier_training_inputs_blocked' if blocked else
            'independent_frontier_training_inputs_passed',
        'failed_checks': [name for name, failed in (
            ('mixed_train_vs_new_reserved', bool(primary_groups)),
            ('mixed_train_vs_full_observed', bool(observed_groups)),
            ('new_reserved_cross_split', any(cross.values())),
            ('cross_split_row_identifier', any(identifier_overlap.values()))) if failed],
        'mixed_train_vs_new_reserved': primary, 'mixed_train_vs_full_observed': exclusions,
        'cross_split_row_identifier_overlap': identifier_overlap,
        'conservative_whole_parent_source_family_fingerprint_counts': dict(sorted(whole_families.items())),
        'new_reserved_cross_split_blocking_fingerprints': cross,
        'typed_clone_confirmation': clones,
        'quarantine_proposal': {'whole_old_train_group_sha256': sorted(digest(g) for g in observed_groups),
            'group_hash_algorithm': 'sha256_canonical_json',
            'whole_parent_rows': sum(len(current[1][g]) for g in observed_groups),
            'whole_parent_groups': len(observed_groups),
            'source_counts': dict(sorted(Counter(r['source'] for g in observed_groups for r in current[1][g]).items())),
            'kind_counts': dict(sorted(Counter(r['kind'] for g in observed_groups for r in current[1][g]).items())),
            'proven_clone_parent_groups': len(proven_groups),
            'proven_clone_parent_rows': sum(len(current[1][g]) for g in proven_groups),
            'additional_conservative_parent_groups': len(observed_groups - proven_groups),
            'additional_conservative_parent_rows': sum(len(current[1][g]) for g in observed_groups - proven_groups)},
        'model_calls': 0,
        'limits': ['Shared magnitude/time atoms are disclosed, not blockers.',
            'Whole-parent abstract-menu equality is a conservative exclusion, not proof every excluded row is contaminated.',
            'Legacy fallback fingerprints do not prove arbitrary paraphrase or graph-isomorphism independence.',
            'Whole-group exclusion does not remove initializer, foundation pretraining or historical exposure.',
            'This input-only audit does not establish model quality or natural/official JevBench performance.']}


def load_inputs(previous, frontier, observed_roots):
    """Privately replay full 1128 and bind selected 904; never interchange them."""
    previous, frontier = Path(previous), Path(frontier)
    watched = {}
    def read(path, checksum):
        raw = checked_bytes(path, checksum); watched[str(path)] = checksum
        return raw
    pinned = {name: json.loads(read(ROOT/name, checksum)) for name, checksum in PINS.items()
              if name.endswith('.json')}
    read(ROOT/'scripts/audit_frontier_isolation_v8.py', PINS['scripts/audit_frontier_isolation_v8.py'])
    old_manifest = pinned['reports/boundary-controls-v7-20261002/prepared-training/manifest.json']
    old_plan = pinned['reports/boundary-controls-v7-20261002/prepared-training/comparison-plan.json']
    freeze = pinned['reports/frontier-v8-data-20261003/freeze-receipt.json']
    read(previous/'manifest.json', PINS['reports/boundary-controls-v7-20261002/prepared-training/manifest.json'])
    read(previous/'comparison-plan.json', PINS['reports/boundary-controls-v7-20261002/prepared-training/comparison-plan.json'])
    read(frontier/'manifest.json', freeze['manifest_sha256'])
    old_raw, old_train = checked_rows(previous/'train.jsonl', old_manifest['files_sha256']['train.jsonl'], 3792, 'train')
    watched[str(previous/'train.jsonl')] = sha(old_raw)
    new_raw, new_rows = {}, {}
    for split in SPLITS:
        checksum = freeze['split_files_sha256'][split+'.jsonl']
        new_raw[split], new_rows[split] = checked_rows(frontier/(split+'.jsonl'), checksum,
            992 if split == 'train' else 496, split, 'frontier-controls-v8')
        watched[str(frontier/(split+'.jsonl'))] = checksum
    observed, original_train, inventory, originals = [], [], {}, {}
    for version in VERSIONS:
        root, identity = Path(observed_roots[version]), old_manifest['configuration']['sources'][version]
        read(root/'manifest.json', identity['manifest_sha256'])
        original = {}
        for split in ('train', 'test', 'ood'):
            checksum = identity['split_sha256'][split+'.jsonl']
            raw, original[split] = checked_rows(root/(split+'.jsonl'), checksum,
                identity['rows'][split], split, VERSIONS[version])
            watched[str(root/(split+'.jsonl'))] = checksum
        original_train.extend(original['train'])
        observed.extend(original['test'] + original['ood'])
        originals[version] = original
        inventory[version] = {'manifest_sha256': identity['manifest_sha256'],
            'test_sha256': identity['split_sha256']['test.jsonl'],
            'ood_sha256': identity['split_sha256']['ood.jsonl'],
            'rows_by_split': {s: len(original[s]) for s in ('test', 'ood')}}
    require(old_train == original_train, 'Previous Train ordered membership/provenance differs from original Train')
    require(len(observed) == 1128, 'Full observed inventory must contain 1128 rows')
    retained_raw, retained_counts = {}, {}
    for version in VERSIONS:
        for split in ('test', 'ood'):
            relative = 'observed-regression/'+version+'/'+split+'.jsonl'
            if version == 'v7':
                path = previous/(split+'.jsonl')
                checksum = old_manifest['files_sha256'][split+'.jsonl']
                selected = originals[version][split]
            else:
                path = previous/relative
                checksum = old_plan['heldout_files_sha256'][relative]
                selected = originals[version][split]
                if version in LOCKS:
                    lock_name, key = LOCKS[version]
                    lock = json.loads(read(ROOT/lock_name, old_plan['observed_regression_locks'][version]['file_sha256']))[key]
                    lookup = {r['id']: r for r in selected}
                    selected = [lookup[identity] for identity in lock['selected_ids'][split]]
                    require(legacy_json_sha(selected) == lock['selected_sha256'][split], 'Observed selection lock order changed')
            raw, rows = checked_rows(path, checksum, len(selected), split, VERSIONS[version])
            watched[str(path)] = checksum
            require(rows == selected, 'Selected observed membership/order differs from original locked subset')
            retained_raw[relative], retained_counts[relative] = raw, len(rows)
    require(sum(retained_counts.values()) == 904, 'Selected observed comparison inventory must contain 904 rows')
    train = old_train + new_rows['train']
    train_summary = safe_schema(train)
    safe_schema(train + [r for s in SPLITS if s != 'train' for r in new_rows[s]])
    bindings = {'previous_manifest_sha256': PINS['reports/boundary-controls-v7-20261002/prepared-training/manifest.json'],
        'previous_plan_sha256': PINS['reports/boundary-controls-v7-20261002/prepared-training/comparison-plan.json'],
        'frontier_manifest_sha256': freeze['manifest_sha256'],
        'frontier_freeze_receipt_sha256': PINS['reports/frontier-v8-data-20261003/freeze-receipt.json'],
        'train_files_sha256': {'prepared_v7': sha(old_raw), 'frontier_v8': sha(new_raw['train'])},
        'new_reserved_files_sha256': {s+'.jsonl': sha(new_raw[s]) for s in SPLITS if s != 'train'},
        'retained_regression_files_sha256': {p: sha(raw) for p, raw in retained_raw.items()},
        'full_observed_inventory': inventory}
    return {'old_train': old_train, 'old_raw': old_raw, 'new_rows': new_rows, 'new_raw': new_raw,
        'observed': observed, 'retained_raw': retained_raw, 'retained_counts': retained_counts,
        'input_bindings': bindings, 'train_summary': train_summary, 'watched': watched}


def audit_inputs(inputs, *, exclusion=None, exclusion_sha256=None, blocked_receipt=None):
    train = inputs['old_train'] + inputs['new_rows']['train']
    reserved = [r for s in SPLITS if s != 'train' for r in inputs['new_rows'][s]]
    original = diagnose(train, reserved, inputs['observed'])
    original.update(schema_version=1, input_bindings=inputs['input_bindings'],
        train_summary=inputs['train_summary'], audit_source_sha256=sha(Path(__file__).read_bytes()),
        primitive_source_sha256=PINS['scripts/audit_frontier_isolation_v8.py'],
        full_observed_rows=len(inputs['observed']), selected_observed_rows=sum(inputs['retained_counts'].values()),
        new_reserved_rows=len(reserved),
        retained_regression_rows=inputs['retained_counts'], artifact_writes=0)
    selected_raw = inputs['old_raw'] + inputs['new_raw']['train']
    original.update(ordered_train_row_ids_sha256=digest([r['id'] for r in train]),
        ordered_train_rows_sha256=digest(train), train_group_ids_sha256=digest(sorted({r['group_id'] for r in train})),
        selected_train_bytes_sha256=sha(selected_raw))
    if exclusion is not None:
        require(blocked_receipt is not None, 'Preserved blocked full-proposal receipt required')
        require(exclusion['status'] == 'independent_whole_parent_exclusion_declared'
            and exclusion['group_hash_algorithm'] == 'sha256_canonical_json', 'Unsupported exclusion declaration')
        require(exclusion['input_bindings'] == original['input_bindings']
            and exclusion['audit_source_sha256'] == original['audit_source_sha256'], 'Exclusion source/input binding differs')
        prior = json.loads(checked_bytes(blocked_receipt, exclusion['blocked_full_preflight_sha256']))
        require(prior == original, 'Preserved full-proposal diagnosis differs from fresh input-only replay')
        expected = original['quarantine_proposal']['whole_old_train_group_sha256']
        require(set(expected) <= {digest(r['group_id']) for r in inputs['old_train']}, 'New frontier data cannot be silently quarantined using observed inputs')
        require(exclusion['excluded_old_train_group_sha256'] == expected, 'Exclusion must be exact unchanged-gate whole-parent closure')
        require(exclusion['excluded_old_train_rows'] == original['quarantine_proposal']['whole_parent_rows'], 'Excluded row count differs')
        excluded = set(expected)
        keep = [digest(r['group_id']) not in excluded for r in inputs['old_train']]
        train = [r for r, use in zip(inputs['old_train'], keep) if use] + inputs['new_rows']['train']
        selected_raw = b''.join(line for line, use in zip(inputs['old_raw'].splitlines(keepends=True), keep) if use) + inputs['new_raw']['train']
        require(safe_schema(train) == exclusion['train_summary'], 'Declared retained ordered membership summary differs')
        result = diagnose(train, reserved, inputs['observed'])
        result.update({k: v for k, v in original.items() if k in ('schema_version', 'audit_source_sha256',
            'primitive_source_sha256', 'full_observed_rows', 'selected_observed_rows', 'new_reserved_rows',
            'retained_regression_rows', 'artifact_writes')})
        require(exclusion_sha256 is not None, 'Exact exclusion declaration bytes hash required')
        result['input_bindings'] = dict(original['input_bindings'], exclusion_declaration_sha256=exclusion_sha256)
        result['train_summary'] = safe_schema(train)
        result['original_full_proposal_status'] = original['status']
        result['applied_whole_parent_exclusion'] = {'excluded_old_groups': len(expected),
            'excluded_old_rows': original['quarantine_proposal']['whole_parent_rows'],
            'proven_clone_parent_groups': original['quarantine_proposal']['proven_clone_parent_groups'],
            'proven_clone_parent_rows': original['quarantine_proposal']['proven_clone_parent_rows'],
            'additional_conservative_parent_groups': original['quarantine_proposal']['additional_conservative_parent_groups'],
            'additional_conservative_parent_rows': original['quarantine_proposal']['additional_conservative_parent_rows']}
    else:
        result = original
    result['ordered_train_row_ids_sha256'] = digest([r['id'] for r in train])
    result['ordered_train_rows_sha256'] = digest(train)
    result['train_group_ids_sha256'] = digest(sorted({r['group_id'] for r in train}))
    result['selected_train_bytes_sha256'] = sha(selected_raw)
    for path, checksum in inputs['watched'].items():
        checked_bytes(path, checksum)
    return result, {s: selected_raw if s == 'train' else inputs['new_raw'][s] for s in SPLITS}


def verify_committed_source(source, audit_sha):
    require(set(source['files_sha256']) == set(SOURCE_FILES)
        and re.fullmatch(r'[0-9a-f]{40}', source['commit']) is not None, 'Stage source inventory differs')
    for name in SOURCE_FILES:
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', source['commit']+':'+name], stderr=subprocess.DEVNULL)
        require(sha(frozen) == source['files_sha256'][name] and (ROOT/name).read_bytes() == frozen,
                'Stage committed implementation binding differs')
    require(source['files_sha256']['scripts/audit_frontier_training_v8.py'] == audit_sha,
            'Stage auditor source differs from independent input receipt')


def verify_stage(dataset, preflight, primary_raw, retained_raw, input_preflight):
    """An optional post-build read-only check cannot waive a blocked preflight."""
    require(preflight['status'] == 'independent_frontier_training_inputs_passed', 'Blocked proposal cannot verify a materialized stage')
    dataset = Path(dataset)
    manifest = json.loads((dataset/'manifest.json').read_bytes())
    receipt_raw = Path(input_preflight).read_bytes()
    require(json.loads(receipt_raw) == preflight and manifest['independent_preflight_sha256'] == sha(receipt_raw),
            'Stage independent preflight binding differs')
    verify_committed_source(manifest['committed_source'], preflight['audit_source_sha256'])
    require(manifest['files_sha256'] == {s+'.jsonl': sha(raw) for s, raw in primary_raw.items()}, 'Stage primary hash inventory differs')
    require(manifest['retained_files_sha256'] == {p: sha(raw) for p, raw in retained_raw.items()}, 'Stage retained inventory differs')
    for split, raw in primary_raw.items():
        require((dataset/(split+'.jsonl')).read_bytes() == raw, 'Stage primary ordered bytes/membership/provenance differs')
    for relative, raw in retained_raw.items():
        require((dataset/relative).read_bytes() == raw, 'Stage retained observed ordered bytes differ')
    require(manifest['input_bindings'] == preflight['input_bindings']
        and manifest['summary'] == preflight['train_summary'], 'Stage input/summary declaration differs')
    membership = manifest['train_membership']
    for key, receipt_key in (('ordered_row_ids_sha256', 'ordered_train_row_ids_sha256'),
            ('ordered_rows_sha256', 'ordered_train_rows_sha256'), ('group_ids_sha256', 'train_group_ids_sha256')):
        require(membership[key] == preflight[receipt_key], 'Stage ordered membership declaration differs')
    require(manifest['materializer_source_sha256'] == manifest['committed_source']['files_sha256']['scripts/prepare_frontier_training_v8.py']
        and manifest['schema_validator_sha256'] == manifest['committed_source']['files_sha256']['jev/data.py'],
        'Stage materializer/schema source declaration differs')
    if 'applied_whole_parent_exclusion' in preflight:
        excluded = manifest['whole_parent_exclusion']
        require(excluded['declaration_sha256'] == preflight['input_bindings']['exclusion_declaration_sha256']
            and excluded['excluded_old_groups'] == preflight['applied_whole_parent_exclusion']['excluded_old_groups']
            and excluded['excluded_old_rows'] == preflight['applied_whole_parent_exclusion']['excluded_old_rows'],
            'Stage whole-parent exclusion declaration differs')
    actual_retained = {p.relative_to(dataset).as_posix() for p in (dataset/'observed-regression').rglob('*.jsonl')}
    require(actual_retained == set(retained_raw), 'Stage observed file membership differs')
    actual_jsonl = {p.relative_to(dataset).as_posix() for p in dataset.rglob('*.jsonl')}
    require(actual_jsonl == {s+'.jsonl' for s in primary_raw} | set(retained_raw), 'Unexpected stage JSONL file')
    return {'status': 'independent_frontier_training_stage_passed', 'manifest_sha256': sha((dataset/'manifest.json').read_bytes()),
        'primary_files_sha256': manifest['files_sha256'], 'retained_files_sha256': manifest['retained_files_sha256'], 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', default=str(ROOT/'data/boundary-training-v7-20261002-r1'))
    parser.add_argument('--frontier', default=str(ROOT/'data/frontier-controls-v8-20261003-r1'))
    parser.add_argument('--exclusion-declaration')
    parser.add_argument('--blocked-full-preflight')
    parser.add_argument('--mixture')
    parser.add_argument('--independent-preflight')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    inputs = load_inputs(args.previous, args.frontier, OBSERVED)
    exclusion_raw = Path(args.exclusion_declaration).read_bytes() if args.exclusion_declaration else None
    exclusion = json.loads(exclusion_raw) if exclusion_raw else None
    result, primary = audit_inputs(inputs, exclusion=exclusion,
        exclusion_sha256=sha(exclusion_raw) if exclusion_raw else None,
        blocked_receipt=args.blocked_full_preflight)
    if exclusion_raw:
        checked_bytes(args.exclusion_declaration, sha(exclusion_raw))
    if args.mixture:
        require(args.independent_preflight is not None, 'Independent input receipt required for stage audit')
        result['stage'] = verify_stage(args.mixture, result, primary, inputs['retained_raw'], args.independent_preflight)
    with Path(args.output).open('x') as output:
        output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'failed_checks', 'train_summary')}, sort_keys=True))
    raise SystemExit(result['status'] != 'independent_frontier_training_inputs_passed')


if __name__ == '__main__':
    main()
