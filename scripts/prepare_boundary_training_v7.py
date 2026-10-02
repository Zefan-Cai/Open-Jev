"""Freeze a new boundary-data candidate; this command never runs a model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

from jev.data import SPLITS, _write_dataset, validate_records
from jev.train import _file_sha256, _json_sha256, source_checkout_commit
from scripts.prepare_policy_training_v6 import LOCKS, RELEASE, ROOT, read_source, write_rows


VERSION = 'boundary-controls-v7'
COUNTS = dict(train=1024, calibration=128, validation=128, test=256, ood=256)
CANDIDATE_LOCK = ROOT/'reports/boundary-controls-v7-20261002/candidate-manifest.json'
VISIBLE_AUDIT = ROOT/'reports/boundary-controls-v7-20261002/visible-overlap.json'
INDEPENDENT_AUDIT = ROOT/'reports/boundary-controls-v7-20261002/independent-audit.json'
IMPLEMENTATION = ('scripts/prepare_boundary_training_v7.py',
    'scripts/prepare_policy_training_v6.py', 'scripts/audit_boundary_controls_v7.py',
    'scripts/screen_boundary_inputs_v7.py', 'scripts/screen_training_overlap.py',
    'scripts/build_boundary_controls_v7.py', 'jev/boundary_controls_v7.py',
    'jev/api.py', 'jev/metrics.py', 'jev/data.py', 'jev/train.py', 'jev/frontier_controls_v4.py',
    'jev/temporal_windows_v5.py', 'jev/policy_controls_v6.py',
    'reports/openjev-hf-data-20261002/original-candidate/candidate-manifest.json',
    RELEASE.relative_to(ROOT).as_posix(),
    *(path.relative_to(ROOT).as_posix() for path, _ in LOCKS.values()),
    CANDIDATE_LOCK.relative_to(ROOT).as_posix(), VISIBLE_AUDIT.relative_to(ROOT).as_posix(),
    INDEPENDENT_AUDIT.relative_to(ROOT).as_posix())


def committed_source():
    commit = source_checkout_commit(__file__)
    for name in IMPLEMENTATION:
        try:
            frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+name],
                                              stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise ValueError('Commit generators, audit and candidate lock before preparing v7') from error
        if frozen != (ROOT/name).read_bytes():
            raise ValueError('Uncommitted preparation input: '+name)
    return commit


def identity_normalized_input(row):
    """Catch identifier-renamed copies while preserving facts and scope relations.

    This narrow screen handles the explicit identity fields in our controls.
    It does not detect arbitrary paraphrases or prove semantic independence.
    """
    state = row['state']
    identities = []
    identity_fields = ('case', 'case_ref', 'account', 'applicant', 'ticket',
                       'submission', 'resource', 'requested_resource', 'department', 'entry_ref')
    if isinstance(state, dict):
        for mapping in (state, state.get('request', {})):
            if isinstance(mapping, dict):
                for key in identity_fields:
                    value = mapping.get(key)
                    if isinstance(value, str) and value and value not in identities:
                        identities.append(value)
    for value in row['metadata'].get('entity_ids', []):
        if value not in identities:
            identities.append(value)
    replacements = sorted(((value, '@identity-'+str(i)+'@') for i, value in enumerate(identities)),
                          key=lambda pair: len(pair[0]), reverse=True)

    unordered = {'ledger', 'signed_events', 'signed_policy_registry', 'events', 'messages', 'policies'}

    def normalize(value, field=None):
        if isinstance(value, str):
            if field in (*identity_fields, 'credential_scope'):
                if value in identities:
                    return '@identity-'+str(identities.index(value))+'@'
                return '@nonmatching-'+field+'@'
            for old, new in replacements:
                value = value.replace(old, new)
            return value
        if isinstance(value, dict):
            return {k: normalize(v, k) for k, v in value.items()}
        if isinstance(value, list):
            values = [normalize(v) for v in value]
            if field in unordered:
                values.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(',', ':')))
            return values
        return value

    return {'state': normalize(state),
            'question': ' '.join(normalize(row['question']).split()), 'kind': row['kind']}


def identity_normalized_context(row):
    return _json_sha256(identity_normalized_input(row))


def reserved_overlap(new_rows, old_rows):
    """Reject both verbatim and identity-renamed copies of any original split."""
    old_contexts = {identity_normalized_context(row) for row in old_rows}
    old_states = {_json_sha256(identity_normalized_input(row)['state']) for row in old_rows}
    old_groups = {row['group_id'] for row in old_rows}
    seen = set()
    for row in new_rows:
        context = identity_normalized_context(row)
        state = _json_sha256(identity_normalized_input(row)['state'])
        if row['group_id'] in old_groups or context in old_contexts or state in old_states:
            raise ValueError('New candidate overlaps an original reserved/source input: '+row['id'])
        if context in seen:
            raise ValueError('Identity-normalized input repeats within the new candidate: '+row['id'])
        seen.add(context)
    return {'status': 'passed', 'new_rows_checked': len(new_rows),
        'reserved_rows_checked': len(old_rows), 'matches': 0,
        'policy': 'All five original v4/v5/v6 splits are reserved against new-copy generation; only their existing Train rows may be reused in the mixture.',
        'limits': 'Explicit identity fields, state reuse and authored unordered lists only; not arbitrary semantic or pretraining independence.'}


def read_candidate(directory):
    # Imported here so metadata/overlap helpers remain usable without a dataset.
    from scripts.audit_boundary_controls_v7 import audit_directory
    directory = Path(directory)
    lock = json.loads(CANDIDATE_LOCK.read_text())
    manifest_bytes = (directory/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if (hashlib.sha256(manifest_bytes).hexdigest() != lock['dataset_manifest_sha256']
            or manifest['files_sha256'] != lock['split_sha256']
            or manifest['generator_sha256'] != lock['generator_sha256']
            or _file_sha256(ROOT/'jev/boundary_controls_v7.py') != lock['generator_sha256']
            or manifest['configuration']['generator_version'] != VERSION):
        raise ValueError('V7 differs from the committed public candidate lock')
    visible = json.loads(VISIBLE_AUDIT.read_text())
    if (_file_sha256(VISIBLE_AUDIT) != lock['visible_overlap_audit_sha256']
            or visible['status'] != 'no_detected_overlap' or visible['matched_rows'] != 0
            or visible['dataset_manifest_sha256'] != lock['dataset_manifest_sha256']
            or visible['split_sha256'] != lock['split_sha256']
            or visible['screen_script_sha256'] != _file_sha256(ROOT/'scripts/screen_boundary_inputs_v7.py')
            or visible['lexical_screen_sha256'] != _file_sha256(ROOT/'scripts/screen_training_overlap.py')
            or visible['compile_api_sha256'] != _file_sha256(ROOT/'jev/api.py')):
        raise ValueError('V7 visible-input overlap receipt differs from the candidate lock')
    audit = audit_directory(directory)
    if (audit['status'] != 'validated_no_model_run' or audit['schema']['splits'] != COUNTS
            or audit['manifest_sha256'] != lock['dataset_manifest_sha256']
            or audit['files_sha256'] != lock['split_sha256']
            or audit['validator_sha256'] != _file_sha256(ROOT/'scripts/audit_boundary_controls_v7.py')):
        raise ValueError('V7 independent audit/counts differ')
    if (_file_sha256(INDEPENDENT_AUDIT) != lock['independent_audit_sha256']
            or json.loads(INDEPENDENT_AUDIT.read_text()) != audit):
        raise ValueError('V7 independent replay differs from its committed public receipt')
    splits = {}
    for split in SPLITS:
        name = split+'.jsonl'
        raw = (directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != lock['split_sha256'][name]:
            raise ValueError('Candidate split changed after independent audit: '+split)
        splits[split] = [json.loads(line) for line in raw.splitlines()]
    if any(row['source'] != VERSION or row['split'] != split
           for split, rows in splits.items() for row in rows):
        raise ValueError('V7 source/split identity differs')
    if ((directory/'manifest.json').read_bytes() != manifest_bytes
            or any(_file_sha256(directory/name) != expected for name, expected in lock['split_sha256'].items())):
        raise ValueError('Candidate bytes changed during preparation')
    return splits, audit


def prepare(v4, v5, v6, v7, output, initial_checkpoint, training_output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh immutable v7 preparation directory')
    commit = committed_source()
    sources, identities = {}, {}
    for name, directory in (('v4', v4), ('v5', v5), ('v6', v6)):
        sources[name], identities[name] = read_source(name, directory)
    sources['v7'], independent = read_candidate(v7)
    identities['v7'] = {'manifest_sha256': independent['manifest_sha256'],
        'split_sha256': independent['files_sha256'],
        'rows_sha256': {s: _json_sha256(rows) for s, rows in sources['v7'].items()},
        'rows': {s: len(rows) for s, rows in sources['v7'].items()},
        'candidate_lock_sha256': _file_sha256(CANDIDATE_LOCK)}
    original = [row for name in ('v4', 'v5', 'v6') for rows in sources[name].values() for row in rows]
    new_rows = [row for rows in sources['v7'].values() for row in rows]
    overlap = reserved_overlap(new_rows, original)
    validate_records(original+new_rows)
    regression, locks = defaultdict(dict), {}
    for name, (path, key) in LOCKS.items():
        selection = json.loads(path.read_text())[key]
        if selection['manifest_sha256'] != identities[name]['manifest_sha256']:
            raise ValueError('Observed regression manifest differs: '+name)
        for split in ('test', 'ood'):
            lookup = {row['id']: row for row in sources[name][split]}
            selected = [lookup[identity] for identity in selection['selected_ids'][split]]
            if _json_sha256(selected) != selection['selected_sha256'][split]:
                raise ValueError('Observed regression selection changed: '+name+'/'+split)
            regression[name][split] = selected
        locks[name] = {'file_sha256': _file_sha256(path),
                       'selected_rows_sha256': selection['selected_sha256']}
    regression['v6'] = {s: sources['v6'][s] for s in ('test', 'ood')}
    mixed = [row for splits in sources.values() for s in ('train', 'calibration', 'validation') for row in splits[s]]
    mixed += sources['v7']['test']+sources['v7']['ood']
    manifest = _write_dataset(mixed, output, {'type': 'synthetic', 'status': 'prepared_not_trained',
        'generator_version': 'boundary-training-v7-mixture', 'seed': 20261003,
        'source_commit': commit, 'sources': identities,
        'train_policy': 'Existing original Train only plus new v7 Train. Observed Test/OOD remain outside Train.'})
    retained = {}
    for name, splits in sources.items():
        for split in ('calibration', 'validation'):
            path = output/'heldout'/name/(split+'.jsonl')
            write_rows(path, splits[split])
            retained[path.relative_to(output).as_posix()] = _file_sha256(path)
    for name, splits in regression.items():
        for split, rows in splits.items():
            path = output/'observed-regression'/name/(split+'.jsonl')
            write_rows(path, rows)
            retained[path.relative_to(output).as_posix()] = _file_sha256(path)
    settings = dict(steps=3792//4, accumulation=4, train_rows=3792,
        calibration_rows=436, eval_rows=256, max_length=4096, lora_rank=8,
        lr=2e-5, head_lr=5e-5, brier_weight=.1, seed=20261003,
        training_sampling='shuffled', checkpoint_every=0)
    if manifest['summary']['splits'] != dict(train=3792, calibration=436, validation=436, test=256, ood=256):
        raise ValueError('Unexpected mixture counts')
    release = json.loads(RELEASE.read_text())
    argv = ['python', '-m', 'jev.train', '--model', release['base_model'], '--revision', release['base_revision'],
            '--data', str(output), '--output', str(training_output), '--initial-checkpoint', str(initial_checkpoint)]
    for key, value in settings.items():
        argv += ['--'+key.replace('_', '-'), str(value)]
    plan = {'schema_version': 1, 'status': 'cpu_data_prepared_comparison_runner_pending',
        'source_commit': commit, 'data_manifest_sha256': _file_sha256(output/'manifest.json'),
        'implementation_sha256': {name: _file_sha256(ROOT/name) for name in IMPLEMENTATION},
        'settings': settings, 'training_argv': argv,
        'expected_initial_checkpoint': {'model_repo': release['model_repository'],
            'model_revision': release['model_repository_revision'], 'directory_sha256': release['checkpoint_sha256'],
            'files_sha256': {n.removeprefix('checkpoint/'): v['sha256'] for n, v in release['files'].items()}},
        'released_package_proof_sha256': _file_sha256(RELEASE),
        'heldout_files_sha256': retained, 'observed_regression_locks': locks,
        'independent_candidate_audit': independent, 'original_source_overlap_audit': overlap,
        'launch_gate': 'Do not launch until a separately committed same-runtime comparison runner and protocol verify this exact plan, new primary and all observed slices, and current GPU ownership/restoration. This preparation never acquires a GPU or runs training.',
        'required_future_comparison': {'reference': 'Exact published 2B versus fixed final v7; v6 is not initialization.',
            'primary': 'New v7 Test/OOD 256 each; all four families, conditions, offsets, numeric magnitude, scope and latest authority separately.',
            'observed': 'Original v4 128+128, v5 32+32, v6 36+36 development regressions; never untouched independent scores.',
            'probabilities': 'Both weights at published and Calibration-only adapted temperatures; argmax, Brier/NLL/ECE and inclusive Noul .2/.8 coverage and accepted errors.',
            'safety': 'For each primary split: zero wrong Choice actions accepting outside windows, missing scope, revoked/missing/withdrawn authority or over-capacity. For each family on pooled Test/OOD: Noul accepted errors=0 and coverage>=0.5 at fixed inclusive .2/.8. Report predicate errors separately from actual execution.',
            'observed_safety': 'All 12 original v5 outside-window Choice rows must reject; all six observed v6 latest-revocation rows must remain correct. Report every previously correct-to-wrong row by family; aggregate gains cannot waive a safety failure.'},
        'stop_policy': 'One shuffled full Train pass, 948 steps, fixed final checkpoint. No Test/OOD tuning, automatic retry, resume or weight promotion. Any interrupted/new attempt must be declared separately.',
        'limits': 'Original synthetic development candidate only. No model result, natural-user result, official JevBench gain or semantic/pretraining-independence proof.'}
    (output/'comparison-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    return manifest, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v4', 'v5', 'v6', 'v7', 'output', 'initial-checkpoint', 'training-output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    manifest, plan = prepare(args.v4, args.v5, args.v6, args.v7, args.output,
                             args.initial_checkpoint, args.training_output)
    print(json.dumps({'status': plan['status'], 'rows': manifest['summary']['splits'],
                      'source_commit': plan['source_commit'], 'steps': plan['settings']['steps']}, indent=2))


if __name__ == '__main__':
    main()
