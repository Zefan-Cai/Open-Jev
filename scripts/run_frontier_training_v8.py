"""Read-only v8 CPU preflight and command construction; launch is unavailable.

A fresh owned launcher/restoration guard is intentionally a separate phase.
No resource-receipt declaration can enable model execution in this driver.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'frontier-v8-isolated-20261003-r1'
NODES = ('ms-n1-1', 'ms-n1-3', 'ms-n4-1', 'ms-n4-2', 'ms-n4-3', 'ms-n4-4')
RUNTIME_NAMES = ('torch', 'transformers', 'peft', 'triton', 'safetensors', 'accelerate')
SETTINGS = {'model': 'Qwen/Qwen3.5-2B', 'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc',
    'steps': 733, 'accumulation': 4, 'train_rows': 2932, 'calibration_rows': 496,
    'eval_rows': 496, 'max_length': 4096, 'lora_rank': 8, 'lr': 2e-5, 'head_lr': 5e-5,
    'brier_weight': .1, 'seed': 20261004, 'training_sampling': 'shuffled',
    'checkpoint_every': 0, 'defer_heldout': True}
NUMERICAL_SETTINGS = {'backbone_dtype': 'torch.bfloat16', 'head_dtype': 'torch.float32',
    'cuda': '12.8', 'float32_matmul_precision': 'highest', 'tf32': False,
    'tf32_scope': 'torch.backends.cuda.matmul.allow_tf32', 'quantization': False}
SOURCE_FILES = ('scripts/run_frontier_training_v8.py', 'jev/train.py', 'jev/model.py',
                'jev/api.py', 'jev/metrics.py', 'jev/data.py')
PATHS = ('source_directory', 'dataset', 'released_checkpoint', 'task_directory',
         'training_run', 'comparison_output', 'completion_receipt', 'runtime_receipt',
         'resource_receipt', 'python', 'hf_cache', 'base_snapshot')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(Path(path).is_file(), 'Regular file required before hashing')
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), 'Regular JSON file required')
    return json.loads(Path(path).read_bytes())


def write_once(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def relative_name(name):
    path = Path(name)
    require(isinstance(name, str) and not path.is_absolute() and '..' not in path.parts
            and path.as_posix() == name and bool(path.parts), 'Invalid relative inventory path')


def inventory(directory, *, cache=None):
    """HF snapshot file links are allowed only into the declared fresh cache."""
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'Regular inventory directory required')
    files, links = {}, {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            require(cache is not None and path.resolve().is_relative_to(Path(cache).resolve())
                    and path.is_file(), 'Input symlink outside verified HF cache')
            links[path.relative_to(directory).as_posix()] = str(path.resolve())
        elif path.is_dir():
            continue
        require(path.is_file(), 'Nonregular inventory leaf rejected')
        files[path.relative_to(directory).as_posix()] = sha(path)
    return files, links


def directory_sha(directory):
    files, _ = inventory(directory)
    result = hashlib.sha256()
    for name in sorted(files):
        result.update(name.encode()+b'\0')
        with (Path(directory)/name).open('rb') as handle:
            for block in iter(lambda: handle.read(8 << 20), b''):
                result.update(block)
    return result.hexdigest()


def git(directory, *arguments):
    environment = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    return subprocess.check_output(['git', '-C', str(directory), *arguments],
                                   env=environment, stderr=subprocess.DEVNULL)


def verify_source(plan, freeze, expected_commit):
    require(re.fullmatch(r'[0-9a-f]{40}', expected_commit) is not None
            and freeze['source_commit'] == expected_commit, 'Frozen source commit differs')
    require(Path(git(ROOT, 'rev-parse', '--show-toplevel').decode().strip()).resolve() == ROOT.resolve()
            and git(ROOT, 'rev-parse', 'HEAD').decode().strip() == expected_commit,
            'Driver checkout differs from frozen source')
    require(not git(ROOT, 'status', '--porcelain', '--untracked-files=all'), 'Dirty source checkout rejected')
    hashes = plan['implementation_sha256']
    require(set(SOURCE_FILES) <= set(hashes) and hashes == freeze['implementation_sha256'],
            'Incomplete or different frozen implementation inventory')
    for name, checksum in hashes.items():
        relative_name(name)
        path = ROOT/name
        require(path.is_file() and not path.is_symlink(), 'Frozen source must be regular')
        raw = git(ROOT, 'show', expected_commit+':'+name)
        require(path.read_bytes() == raw and hashlib.sha256(raw).hexdigest() == checksum,
                'Changed/uncommitted frozen implementation: '+name)
    return hashes


def validate_paths(request, request_path, *, claimed=False):
    require(type(request.get('schema_version')) is int and request['schema_version'] == 1
            and request.get('protocol_id') == PROTOCOL
            and re.fullmatch(r'frontier-v8-[0-9a-f]{16,64}', request.get('attempt_id', '')) is not None,
            'Fresh v8 attempt identity required')
    require(request.get('node_alias') in NODES, 'Node outside authorized six-node inventory')
    for name in PATHS:
        path = Path(request[name])
        require(path.is_absolute() and not any(word in str(path).lower()
            for word in ('future-host', 'placeholder', '<', '>')), 'Resolve actual host path: '+name)
    task = Path(request['task_directory'])
    require(task.is_dir() and not task.is_symlink() and task.name == request['attempt_id'],
            'Fresh regular attempt directory required')
    expected = {'training_run': 'training', 'comparison_output': 'comparison',
        'completion_receipt': 'training-completion.json', 'runtime_receipt': 'cpu-stage-receipt.json',
        'resource_receipt': 'resource-ready.json'}
    for name, child in expected.items():
        require(Path(request[name]) == task/child, 'Attempt path is not the fixed fresh child: '+name)
    require(Path(request_path) == task/'execution-request.json', 'Request must be in its fresh task root')
    inputs = [Path(request[name]).resolve() for name in
              ('source_directory', 'dataset', 'released_checkpoint', 'hf_cache')]
    for source in inputs:
        require(not task.resolve().is_relative_to(source) and not source.is_relative_to(task.resolve()),
                'Attempt directory overlaps immutable input')
    for name in ('training_run', 'comparison_output', 'completion_receipt'):
        path = Path(request[name])
        require(not path.exists() and not path.is_symlink(), 'Existing attempt output rejected')
    require(claimed or not (task/'attempt.lock.json').exists(), 'Attempt already consumed; no retry')
    require(not (task/'attempt-failure.json').exists(), 'Failed attempt cannot be reused')
    allowed = {'execution-request.json', 'cpu-stage-receipt.json', 'resource-ready.json',
               'cpu-preflight.json', 'protocol-freeze.json'} | ({'attempt.lock.json'} if claimed else set())
    require(all(p.name in allowed and p.is_file() and not p.is_symlink() for p in task.iterdir()),
            'Unexpected/old attempt artifact or symlink in fresh task root')
    return task


def runtime_identity(lock):
    """Read first-match distributions and import positions without importing Torch."""
    pins = {}
    for line in Path(lock).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        require(match is not None and match[1] not in pins, 'Invalid/duplicate runtime lock entry')
        pins[match[1]] = match[2]
    versions, roots = {}, {}
    for name, expected in pins.items():
        distribution = metadata.distribution(name)
        versions[name] = distribution.version
        roots[name] = str(Path(distribution.locate_file('')).resolve())
        require(versions[name] == expected, 'Installed runtime lock version differs: '+name)
    origins = {}
    for name in RUNTIME_NAMES:
        spec = util.find_spec(name)
        require(spec is not None and spec.origin is not None, 'Missing runtime package position: '+name)
        origins[name] = str(Path(spec.origin).resolve())
    return {'versions': versions, 'distribution_roots': roots, 'package_origins': origins}


def translated_argv(plan, request):
    require(plan['settings'] == SETTINGS, 'Training settings differ from fixed one-pass declaration')
    command = [request['python'], '-m', 'jev.train']
    for name, value in SETTINGS.items():
        if name == 'defer_heldout':
            continue
        command.extend(['--'+name.replace('_', '-'), str(value)])
    command.extend(['--data', request['dataset'], '--output', request['training_run'],
                    '--initial-checkpoint', request['released_checkpoint'], '--defer-heldout'])
    return command


def prepare(args, *, claimed=False):
    request_path = Path(args.request)
    request = read_json(request_path)
    validate_paths(request, request_path, claimed=claimed)
    require(sha(args.plan) == args.expected_plan_sha256 and sha(args.freeze) == args.expected_freeze_sha256,
            'Exact public plan/freeze bytes required')
    plan, freeze = read_json(args.plan), read_json(args.freeze)
    require(type(plan['schema_version']) is type(freeze['schema_version']) is int
            and plan['schema_version'] == freeze['schema_version'] == 1 and plan['protocol_id'] == PROTOCOL
            and plan['settings'] == SETTINGS and plan['settings']['defer_heldout'] is True
            and plan['runtime_settings'] == NUMERICAL_SETTINGS
            and plan['runtime_settings']['tf32'] is False and plan['runtime_settings']['quantization'] is False
            and plan['resource_protocol_ready'] is False
            and plan['gpu_authority'] is False, 'Only the CPU v8 protocol declaration is supported')
    require(freeze['status'] == 'frontier_v8_protocol_frozen_to_committed_source'
            and freeze['source_pushed_before_freeze'] is True
            and freeze['plan_sha256'] == args.expected_plan_sha256
            and freeze['data_manifest_sha256'] == plan['data_manifest_sha256'], 'Protocol freeze binding differs')
    require(request['source_commit'] == args.expected_commit
            and Path(request['source_directory']).resolve() == ROOT.resolve(), 'Request source checkout differs')
    source = verify_source(plan, freeze, args.expected_commit)
    data = Path(request['dataset'])
    require(sha(data/'manifest.json') == plan['data_manifest_sha256'], 'Prepared data manifest changed')
    files, _ = inventory(data)
    require(files == {**plan['data_files_sha256'], 'manifest.json': plan['data_manifest_sha256']},
            'Prepared dataset inventory/content changed')
    manifest = read_json(data/'manifest.json')
    require({**manifest['files_sha256'], **manifest['retained_files_sha256']} == plan['data_files_sha256'],
            'Prepared split inventory differs')
    from jev.data import validate_records
    from jev.train import initial_checkpoint_identity, read_rows, OPTIMIZER_SETTINGS
    require(plan['optimizer'] == OPTIMIZER_SETTINGS, 'Frozen optimizer declaration differs from trainer')
    train = read_rows(data/'train.jsonl', 0, SETTINGS['seed'], balanced=False)
    calibration = read_rows(data/'calibration.jsonl', 0, SETTINGS['seed'], balanced=True)
    validate_records([*train, *calibration])
    require(len(train) == len({r['id'] for r in train}) == 733*4
            and len(calibration) == len({r['id'] for r in calibration}) == 496,
            'Complete Train/Calibration membership differs')
    profile = SimpleNamespace(**SETTINGS, initial_checkpoint=request['released_checkpoint'],
                              output=request['training_run'], resume_training=None)
    initial = initial_checkpoint_identity(profile)
    expected = plan['expected_initial_checkpoint']
    checkpoint_files, _ = inventory(request['released_checkpoint'])
    require(expected['model_repo'] == 'ZefanCai/Open-Jev-2B'
            and expected['model_revision'] == '0c7aa498b1627be8da4acf34c863ff0ee0a92785'
            and initial['files_sha256'] == checkpoint_files == expected['files_sha256']
            and directory_sha(request['released_checkpoint']) == expected['directory_sha256']
            and initial['temperature'] == plan['published_temperature'], 'Exact published initializer required')
    lock = ROOT/plan['runtime_lock']['path']
    require(not lock.is_symlink() and sha(lock) == plan['runtime_lock']['sha256'], 'Frozen runtime lock changed')
    target = plan['runtime_lock']
    require((platform.system(), platform.machine(), f'{sys.version_info.major}.{sys.version_info.minor}')
            == (target['platform'], target['machine'], target['python']), 'Runtime platform/Python differs')
    runtime = runtime_identity(lock)
    require(len(runtime['versions']) == target['distributions']
            and {n: runtime['versions'][n] for n in RUNTIME_NAMES} == plan['runtime'] == request['runtime'],
            'Runtime distribution inventory differs')
    require(Path(request['python']).resolve() == Path(sys.executable).resolve(), 'Different training interpreter')
    stage_path = Path(request['runtime_receipt'])
    require(sha(stage_path) == request['runtime_receipt_sha256'], 'Fresh CPU staging receipt changed')
    stage = read_json(stage_path)
    require(stage['schema_version'] == 1 and stage['status'] == 'cpu_staged_no_cuda_initialization'
            and stage['cuda_initialized'] is False and stage['protocol_id'] == PROTOCOL
            and stage['attempt_id'] == request['attempt_id'] and stage['source_commit'] == args.expected_commit
            and stage['plan_sha256'] == args.expected_plan_sha256
            and stage['python'] == request['python'] and stage['runtime'] == plan['runtime']
            and stage['hf_cache'] == request['hf_cache'] and stage['base_snapshot'] == request['base_snapshot']
            and stage['runtime_identity'] == runtime and stage['python_sha256'] == sha(request['python']),
            'CPU stage source/runtime/attempt binding differs')
    cache, snapshot = Path(request['hf_cache']), Path(request['base_snapshot'])
    require(cache.is_dir() and not cache.is_symlink() and snapshot.resolve().is_relative_to(cache.resolve())
            and snapshot.name == SETTINGS['revision'], 'Pinned base snapshot/cache path differs')
    base, links = inventory(snapshot, cache=cache)
    require(base == plan['expected_base_snapshot_files_sha256'] == stage['base_snapshot_files_sha256']
            and links == stage['base_snapshot_symlink_targets'], 'Pinned base content/link targets changed')
    overlay = stage.get('overlay')
    if overlay:
        require(Path(overlay).is_absolute() and inventory(overlay)[0] == stage['overlay_files_sha256'],
                'Runtime overlay content changed')
    else:
        require(not stage.get('overlay_files_sha256'), 'Overlay inventory without overlay')
    for name, origin in runtime['package_origins'].items():
        allowed = [Path(runtime['distribution_roots'][name])]+([Path(overlay).resolve()] if overlay else [])
        require(any(Path(origin).is_relative_to(path) for path in allowed), 'Package shadows staged runtime: '+name)
    return request, plan, {'schema_version': 1, 'status': 'cpu_preflight_passed_launch_unavailable',
        'protocol_id': PROTOCOL, 'attempt_id': request['attempt_id'], 'source_commit': args.expected_commit,
        'source_sha256': source, 'plan_sha256': args.expected_plan_sha256,
        'freeze_receipt_sha256': args.expected_freeze_sha256, 'request_sha256': sha(request_path),
        'runtime_receipt_sha256': sha(stage_path), 'runtime': plan['runtime'],
        'required_numerical_settings': NUMERICAL_SETTINGS, 'numerical_settings_applied': False,
        'runtime_identity': runtime, 'data_files_sha256': files, 'initial_checkpoint': initial,
        'dataset': request['dataset'],
        'train_selection_sha256': json_sha([r['id'] for r in train]),
        'calibration_selection_sha256': json_sha([r['id'] for r in calibration]),
        'train_rows': len(train), 'calibration_rows': len(calibration),
        'resolved_training_argv': translated_argv(plan, request),
        'model_calls': 0, 'GPU_actions': 0, 'resource_protocol_ready': False, 'execution_available': False,
        'limits': ['This preflight hashes heldout bytes but parses only Train/Calibration.',
                   'No fresh owned launcher/restoration guard exists in this CPU phase; execution remains unavailable.']}


def execute(args):
    """Consume a valid fresh attempt; declaration flags cannot unlock execution."""
    request = read_json(args.request)
    task = validate_paths(request, Path(args.request))
    write_once(task/'attempt.lock.json', {'schema_version': 1, 'protocol_id': PROTOCOL,
        'attempt_id': request['attempt_id'], 'status': 'attempt_consumed_no_retry',
        'claimed_at_utc': datetime.now(timezone.utc).isoformat(), 'model_calls': 0})
    try:
        prepare(args, claimed=True)
        raise ValueError('Execution unavailable: separately reviewed fresh owned launcher/restoration guard required')
    except BaseException as error:
        write_once(task/'attempt-failure.json', {'schema_version': 1,
            'status': 'attempt_failed_consumed_no_retry', 'exception_type': type(error).__name__,
            'model_calls': 0, 'GPU_actions': 0, 'automatic_promotion': False})
        raise


def validate_completed_run(run, plan, preflight):
    """CPU validation for a future owned launcher's fixed-final completion."""
    run = Path(run)
    inventory(run)  # Reject special leaves and symlinks before opening journals.
    require(not (run/'training-checkpoints').exists()
            and not any((run/name).exists() for name in
                ('baseline_test.jsonl', 'baseline_ood.jsonl', 'baseline_calibration.jsonl',
                 'trained_test.jsonl', 'trained_ood.jsonl')), 'Resume snapshot or heldout artifacts forbidden')
    meta, summary = read_json(run/'run.json'), read_json(run/'summary.json')
    require(summary['status'] == 'complete' and summary['steps'] == SETTINGS['steps']
            and summary['trained_rows_consumed'] == SETTINGS['train_rows']
            and summary['checkpoint_selection'] == 'fixed_final_step'
            and summary['checkpoint_reload_split'] == 'calibration' and summary['metrics'] == {}
            and summary['baseline_temperature'] is None
            and meta['commit'] == preflight['source_commit'] and not meta.get('resume_training')
            and meta['initial_checkpoint_identity'] == preflight['initial_checkpoint']
            and meta['baseline_initialization'] == 'inference_checkpoint'
            and meta['phase'] == 'complete'
            and all(meta.get(k) == v for k, v in SETTINGS.items()), 'Fixed-final training declaration differs')
    require(meta['heldout_evaluation'] == summary['heldout_evaluation'] ==
            {'status': 'deferred', 'splits': ['validation', 'test', 'ood'], 'model_calls': 0}
            and meta['evaluation_ids'] == meta['ood_ids'] == []
            and meta['data_sha256'] == {s: plan['data_files_sha256'][s+'.jsonl']
                                      for s in ('train', 'calibration')}, 'Heldout boundary or Train/Cal binding differs')
    journal = [json.loads(line) for line in (run/'training.jsonl').read_bytes().splitlines()]
    require([r['step'] for r in journal] == list(range(1, 734)), 'Incomplete/duplicate optimizer journal')
    for record in journal:
        for name in ('loss', 'gradient_norm', 'elapsed_seconds', 'peak_memory_gib'):
            value = record.get(name)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    'Invalid/nonfinite training journal field')
    require(all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a, b in zip(journal, journal[1:])),
            'Nonmonotonic optimizer elapsed time')
    calibration = [json.loads(line) for line in (run/'calibration.jsonl').read_bytes().splitlines()]
    from jev.train import read_rows, initial_checkpoint_identity
    calibration_path = Path(preflight['dataset'])/'calibration.jsonl'
    require(sha(calibration_path) == plan['data_files_sha256']['calibration.jsonl'], 'Calibration data changed')
    rows = read_rows(calibration_path, 0, SETTINGS['seed'], balanced=True)
    ids = [r['id'] for r in calibration]
    require(len(ids) == len(set(ids)) == 496 and ids == meta['calibration_ids']
            and ids == [r['id'] for r in rows]
            and json_sha(ids) == preflight['calibration_selection_sha256'], 'Calibration journal identity differs')
    for record, row in zip(calibration, rows):
        require(all(record.get(k) == row[k] for k in ('id', 'group_id', 'source', 'kind', 'target'))
                and len(record['logits']) == len(row['options'])
                and all(type(v) in (int, float) and math.isfinite(v) for v in record['logits']),
                'Calibration logits invalid')
    saved = read_json(run/'checkpoint/temperature.json')
    initial_checkpoint_identity(SimpleNamespace(**SETTINGS, initial_checkpoint=str(run/'checkpoint'),
        output=str(run.parent/'comparison'), resume_training=None))
    from jev.metrics import fit_temperature
    fitted = fit_temperature([r['logits'] for r in calibration], [r['target'] for r in calibration])
    require(saved['split'] == 'calibration' and saved['n'] == 496
            and saved['ids_sha256'] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            and saved['temperature'] == summary['temperature']
            and math.isclose(fitted, saved['temperature'], rel_tol=1e-12, abs_tol=1e-12),
            'Temperature not bound to completed Calibration journal')
    error = summary['checkpoint_reload_max_error']
    require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= .05,
            'Invalid fixed-final checkpoint reload error')
    reload_rows = [json.loads(line) for line in (run/'reload_check.jsonl').read_bytes().splitlines()]
    require(len(reload_rows) == 1 and all(reload_rows[0].get(k) == rows[0][k]
            for k in ('id', 'group_id', 'source', 'kind', 'target'))
            and len(reload_rows[0]['logits']) == len(rows[0]['options'])
            and all(type(v) in (int, float) and math.isfinite(v) for v in reload_rows[0]['logits']),
            'Calibration reload journal differs')
    measured_error = max(abs(a-b) for a, b in zip(calibration[0]['logits'], reload_rows[0]['logits']))
    require(math.isclose(measured_error, error, rel_tol=1e-12, abs_tol=1e-12), 'Claimed reload error differs')
    phases = summary['phase_metrics']
    require(phases == meta['phase_metrics'] and set(phases) == {'model_loading', 'warmup', 'optimizer',
        'checkpoint_save', 'calibration_and_temperature', 'checkpoint_reload'}, 'Six phase telemetry records required')
    for phase in phases.values():
        require(set(phase) == {'elapsed_seconds', 'peak_allocated_tensor_memory_gib',
                              'peak_reserved_allocator_memory_gib'}
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in phase.values())
                and phase['peak_allocated_tensor_memory_gib'] <= phase['peak_reserved_allocator_memory_gib'],
                'Invalid allocated/reserved tensor telemetry')
    return {'completed_steps': 733, 'consumed_rows': 2932,
        'artifacts_sha256': {name: sha(run/name) for name in
            ('run.json', 'summary.json', 'training.jsonl', 'calibration.jsonl', 'reload_check.jsonl')},
        'checkpoint': {'files_sha256': inventory(run/'checkpoint')[0],
                       'directory_sha256': directory_sha(run/'checkpoint')},
        'calibration': {'journal_sha256': sha(run/'calibration.jsonl'), 'ordered_ids_sha256': json_sha(ids),
                        'count': 496, 'temperature': saved['temperature']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute', action='store_true', help='Unavailable until a new owned launcher is reviewed')
    args = parser.parse_args()
    if args.execute:
        return execute(args)
    _, _, receipt = prepare(args)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
