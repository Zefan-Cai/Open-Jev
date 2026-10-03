"""Byte-preserving provisional v8 mixture; independent preflight is mandatory."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

from jev.data import validate_records


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT/'scripts/audit_frontier_training_v8.py'
PUBLIC = {
    'previous_manifest': (ROOT/'reports/boundary-controls-v7-20261002/prepared-training/manifest.json',
                          'adcf3c66af2c6398c9039220172d10ce3398c455f40d7fc9bfc0af639a33aae0'),
    'previous_plan': (ROOT/'reports/boundary-controls-v7-20261002/prepared-training/comparison-plan.json',
                      'bab40284419d8ff0bfcdca818047cc96581d0d4e69a7fb891f1c08fe0ec49f35'),
    'frontier_manifest': (ROOT/'reports/frontier-v8-data-20261003/manifest.json',
                          'dd8a9d941c2a73b9ac222bfeee53c4284f854a89fde1496ace6531e318244fd9'),
    'frontier_freeze_receipt': (ROOT/'reports/frontier-v8-data-20261003/freeze-receipt.json',
                                'cd554fe00bd592febdf01a0b41c20de676054e54484a1bd27dc8f86ee4e20232'),
}
SUMMARY_KEYS = ('records', 'groups', 'unique_inputs', 'splits', 'kinds', 'families', 'sources')
RESERVED = ('calibration', 'validation', 'test', 'ood')
SOURCE_FILES = ('scripts/prepare_frontier_training_v8.py', 'scripts/audit_frontier_training_v8.py',
                'jev/train.py', 'jev/data.py', 'jev/api.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def bound_bytes(path, expected):
    path = Path(path)
    require(not path.is_symlink(), 'Input symlink rejected: '+str(path))
    raw = path.read_bytes()
    require(sha(raw) == expected, 'Locked input hash differs before parsing/copying: '+str(path))
    return raw


def committed_source():
    """Require only the explicitly bound source files clean in HEAD."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith('GIT_')}
    try:
        top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
                                      env=environment, text=True).strip()
        require(Path(top).resolve() == ROOT.resolve(), 'Git checkout differs from materializer source root')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                         env=environment, text=True).strip()
        files = {}
        for name in SOURCE_FILES:
            recorded = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT,
                                               env=environment, stderr=subprocess.DEVNULL)
            actual = (ROOT/name).read_bytes()
            require(actual == recorded, 'Uncommitted materializer/auditor/trainer source: '+name)
            files[name] = sha(actual)
    except (subprocess.CalledProcessError, OSError) as error:
        raise ValueError('Materializer/auditor/trainer sources must be committed') from error
    return {'commit': commit, 'files_sha256': files}


def load_lock():
    documents = {name: json.loads(bound_bytes(path, digest))
                 for name, (path, digest) in PUBLIC.items()}
    previous, plan = documents['previous_manifest'], documents['previous_plan']
    frontier, freeze = documents['frontier_manifest'], documents['frontier_freeze_receipt']
    require(plan['data_manifest_sha256'] == PUBLIC['previous_manifest'][1], 'Previous manifest binding differs')
    require(freeze['status'] == 'original_v8_dataset_frozen_to_local_committed_source'
            and not freeze['failed_checks'] and freeze['manifest_sha256'] == PUBLIC['frontier_manifest'][1]
            and freeze['split_files_sha256'] == frontier['files_sha256'], 'Original v8 freeze binding differs')
    require(previous['configuration']['generator_version'] == 'boundary-training-v7-mixture'
            and frontier['configuration']['generator_version'] == 'frontier-controls-v8',
            'Only the exact declared original sources are allowed')
    retained = {f'observed-regression/{origin}/{split}.jsonl':
                plan['heldout_files_sha256'][f'observed-regression/{origin}/{split}.jsonl']
                for origin in ('v4', 'v5', 'v6') for split in ('test', 'ood')}
    retained.update({f'observed-regression/v7/{split}.jsonl': previous['files_sha256'][split+'.jsonl']
                     for split in ('test', 'ood')})
    full = {}
    for origin in ('v4', 'v5', 'v6', 'v7'):
        source = previous['configuration']['sources'][origin]
        full[origin] = {'manifest_sha256': source['manifest_sha256'],
            'test_sha256': source['split_sha256']['test.jsonl'],
            'ood_sha256': source['split_sha256']['ood.jsonl'],
            'rows_by_split': {split: source['rows'][split] for split in ('test', 'ood')}}
    sources = {'frontier-controls-v4': 2400, 'temporal-windows-v5': 128,
               'original-policy-controls-v6-candidate': 240, 'boundary-controls-v7': 1024}
    return {'input_bindings': {
        **{name+'_sha256': digest for name, (_, digest) in PUBLIC.items()},
        'train_files_sha256': {'prepared_v7': previous['files_sha256']['train.jsonl'],
                              'frontier_v8': frontier['files_sha256']['train.jsonl']},
        'new_reserved_files_sha256': {split+'.jsonl': frontier['files_sha256'][split+'.jsonl'] for split in RESERVED},
        'retained_regression_files_sha256': retained, 'full_observed_inventory': full},
        'source_counts': {'prepared_v7': sources, 'frontier_v8': {'frontier-controls-v8': 992}},
        'expected_summary': {'records': 4784, 'groups': 834, 'unique_inputs': 4784,
            'splits': {'train': 4784}, 'kinds': {'choice': 3084, 'noul': 1700},
            'families': {'policy': 4288, 'routing': 496},
            'sources': {**sources, 'frontier-controls-v8': 992}}}


def validate_train_blocks(blocks, lock):
    """Validate only Train; content and ancestor membership remain unchanged."""
    rows, owners = [], {}
    for origin, block in blocks.items():
        require(bool(block) and all(row['split'] == 'train' for row in block),
                'Reserved or misassigned split cannot enter Train')
        require(dict(Counter(row['source'] for row in block)) == lock['source_counts'][origin],
                'Train source ancestry differs: '+origin)
        for row in block:
            group = row['group_id']
            require(group not in owners or owners[group] == origin, 'Parent shared between Train origins')
            owners[group] = origin
        rows.extend(block)
    summary = validate_records(rows)
    summary = {key: summary[key] for key in SUMMARY_KEYS}
    require(summary['unique_inputs'] == len(rows), 'Repeated model-visible Train input')
    require(summary == lock['expected_summary'], 'Complete immutable Train membership/counts differ')
    return rows, summary


def verify_preflight(path, bindings, summary):
    path = Path(path)
    require(not path.is_symlink(), 'Preflight receipt symlink rejected')
    raw = path.read_bytes()
    receipt = json.loads(raw)
    require(type(receipt.get('schema_version')) is int and receipt.get('schema_version') == 1
            and receipt.get('status') == 'independent_frontier_training_inputs_passed'
            and type(receipt.get('model_calls')) is int and receipt.get('model_calls') == 0
            and receipt.get('failed_checks') == [],
            'Independent input isolation did not pass; no materialization')
    require(receipt.get('audit_source_sha256') == sha(AUDITOR.read_bytes()), 'Preflight auditor source differs')
    require(receipt.get('input_bindings') == bindings and receipt.get('train_summary') == summary,
            'Preflight is not bound to this exact Train/reserved inventory')
    return sha(raw)


def select_declared_parents(path, bindings, blocks, raw_train):
    """Apply only an explicitly audited complete-old-parent declaration."""
    raw = Path(path).read_bytes()
    declaration = json.loads(raw)
    require(declaration.get('schema_version') == 1
            and declaration.get('status') == 'independent_whole_parent_exclusion_declared'
            and declaration.get('audit_source_sha256') == sha(AUDITOR.read_bytes())
            and declaration.get('input_bindings') == bindings,
            'Whole-parent exclusion declaration is not bound to these original inputs')
    excluded = declaration.get('excluded_old_train_group_sha256')
    require(type(excluded) is list and len(excluded) == len(set(excluded)) == 335,
            'Explicit unique whole-parent exclusion hashes required')
    require(declaration.get('group_hash_algorithm') == 'sha256_canonical_json', 'Unreviewed parent hash algorithm')
    excluded = set(excluded)
    old = blocks['prepared_v7']
    require(excluded <= {json_sha(row['group_id']) for row in old}
            and not excluded & {json_sha(row['group_id']) for row in blocks['frontier_v8']},
            'Exclusions must name only original old Train parents')
    keep = [json_sha(row['group_id']) not in excluded for row in old]
    require(keep.count(False) == declaration.get('excluded_old_train_rows') == 1852,
            'Declared whole-parent removed-row count differs')
    require(type(declaration.get('blocked_full_preflight_sha256')) is str
            and len(declaration['blocked_full_preflight_sha256']) == 64,
            'Preserved blocked full-input receipt binding required')
    selected = {**blocks, 'prepared_v7': [row for row, retained in zip(old, keep) if retained]}
    selected_raw = {**raw_train, 'prepared_v7': b''.join(line for line, retained in
        zip(raw_train['prepared_v7'].splitlines(keepends=True), keep) if retained)}
    rows = selected['prepared_v7']+selected['frontier_v8']
    summary = validate_records(rows)
    summary = {key: summary[key] for key in SUMMARY_KEYS}
    require(summary == declaration.get('train_summary') and summary['records'] == 2932
            and summary['groups'] == 499 and summary['unique_inputs'] == 2932
            and summary['kinds'] == {'choice': 1951, 'noul': 981},
            'Declared derivative membership/counts differ')
    evidence = {'declaration_sha256': sha(raw), 'blocked_full_preflight_sha256': declaration['blocked_full_preflight_sha256'],
        'excluded_old_groups': len(excluded), 'excluded_old_rows': keep.count(False),
        'original_full_train_rows': 4784,
        'interpretation': 'Whole-parent exclusions include conservative abstract menus; excluded parents are not all proven contamination.'}
    return selected, selected_raw, rows, summary, evidence


def prepare(previous, frontier, output, independent_preflight, exclusion_declaration=None):
    previous, frontier, output = map(Path, (previous, frontier, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError('Exclusive provisional output already exists: '+str(output))
    for source in (previous, frontier):
        require(source.is_dir() and not source.is_symlink(), 'Original source directory required')
        require(not output.resolve().is_relative_to(source.resolve())
                and not source.resolve().is_relative_to(output.resolve()), 'Output overlaps immutable source')
    source = committed_source()
    lock = load_lock()
    bindings = lock['input_bindings']
    bound_bytes(previous/'manifest.json', bindings['previous_manifest_sha256'])
    bound_bytes(previous/'comparison-plan.json', bindings['previous_plan_sha256'])
    bound_bytes(frontier/'manifest.json', bindings['frontier_manifest_sha256'])
    raw_train = {origin: bound_bytes(directory/'train.jsonl', bindings['train_files_sha256'][origin])
                 for origin, directory in (('prepared_v7', previous), ('frontier_v8', frontier))}
    require(all(raw.endswith(b'\n') for raw in raw_train.values()), 'Train blocks require final newline for exact concatenation')
    blocks = {origin: [json.loads(line) for line in raw.splitlines()] for origin, raw in raw_train.items()}
    rows, summary = validate_train_blocks(blocks, lock)
    exclusion = None
    if exclusion_declaration is not None:
        blocks, raw_train, rows, summary, exclusion = select_declared_parents(
            exclusion_declaration, bindings, blocks, raw_train)
        bindings = {**bindings, 'exclusion_declaration_sha256': exclusion['declaration_sha256']}
    # A blocked receipt fails BEFORE any reserved byte read or output creation.
    preflight_sha = verify_preflight(independent_preflight, bindings, summary)
    copies = {name: bound_bytes(frontier/name, digest)
              for name, digest in bindings['new_reserved_files_sha256'].items()}
    retained = {}
    for name, digest in bindings['retained_regression_files_sha256'].items():
        original = previous/Path(name).name if name.startswith('observed-regression/v7/') else previous/name
        retained[name] = bound_bytes(original, digest)
    train = raw_train['prepared_v7']+raw_train['frontier_v8']
    manifest = {'schema_version': 1, 'status': 'provisional_materialized_not_training_plan_frozen',
        'configuration': {'type': 'synthetic', 'generator_version': 'frontier-training-v8-mixture',
            'train_order': ['prepared_v7', 'frontier_v8'], 'metadata_or_id_rewrites': False},
        'summary': summary,
        'committed_source': source, 'whole_parent_exclusion': exclusion,
        'input_bindings': bindings, 'independent_preflight_sha256': preflight_sha,
        'materializer_source_sha256': sha(Path(__file__).read_bytes()),
        'schema_validator_sha256': sha((ROOT/'jev/data.py').read_bytes()),
        'files_sha256': {'train.jsonl': sha(train), **{name: sha(raw) for name, raw in copies.items()}},
        'retained_files_sha256': {name: sha(raw) for name, raw in retained.items()},
        'train_membership': {'ordered_row_ids_sha256': json_sha([row['id'] for row in rows]),
            'ordered_rows_sha256': json_sha(rows), 'group_ids_sha256': json_sha(sorted({row['group_id'] for row in rows})),
            'blocks': {origin: {'rows': len(block), 'groups': len({row['group_id'] for row in block}),
                'ordered_row_ids_sha256': json_sha([row['id'] for row in block]),
                'ordered_rows_sha256': json_sha(block)} for origin, block in blocks.items()}},
        'reserved_byte_copy_only': True,
        'reserved_counts_previously_audited_not_parsed_here': {split: 496 for split in RESERVED},
        'selected_observed_rows': {'v4': {'test': 128, 'ood': 128}, 'v5': {'test': 32, 'ood': 32},
            'v6': {'test': 36, 'ood': 36}, 'v7': {'test': 256, 'ood': 256}},
        'selected_observed_total': 904, 'full_observed_screen_total': 1128,
        'post_materialization_independent_audit': 'required_before_any_mixture_or_training_plan_freeze',
        'training_mixture_frozen': False, 'initializer_frozen': False, 'steps_frozen': False,
        'comparison_protocol_frozen': False, 'model_calls': 0, 'training_runs': 0,
        'limits': ['Preflight verifies input isolation; a separate independent audit must verify actual output bytes and order.',
                   'Old observed scores are development feedback; the selected904 inventory differs from the full1128 overlap screen.',
                   'This artifact contains no executable argv, model initializer, optimizer settings, run or GPU authorization.']}
    output.mkdir(parents=True)
    for name, raw in {'train.jsonl': train, **copies, **retained}.items():
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)
    with (output/'manifest.json').open('x') as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=ROOT/'data/boundary-training-v7-20261002-r1')
    parser.add_argument('--frontier', type=Path, default=ROOT/'data/frontier-controls-v8-20261003-r1')
    parser.add_argument('--output', type=Path, default=ROOT/'data/frontier-training-v8-20261003-provisional-r1')
    parser.add_argument('--independent-preflight', type=Path, required=True)
    parser.add_argument('--exclusion-declaration', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.previous, args.frontier, args.output, args.independent_preflight, args.exclusion_declaration)
    print(json.dumps({'status': report['status'], 'train': report['summary']['records'], 'model_calls': 0}))


if __name__ == '__main__':
    main()
