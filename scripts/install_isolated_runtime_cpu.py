"""One bounded CPU-only installation of the frozen Linux runtime in a new venv.

This utility creates no model, GPU job or launch authority. Failed directories
and consumed markers remain as evidence. It never changes the bootstrap env.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA256 = 'd1f238a916dfdfbc8b89f3de4c28bb3b7220cdd97ecf609ebb726afab0ca8221'
NAMES = ('torch', 'transformers', 'peft', 'triton', 'safetensors', 'accelerate')
MAX_SECONDS = 1800
_ACTIVE_INTERRUPTION = None
PROBE = r'''
import importlib.metadata as metadata, importlib.util as util, json, site, sys
from pathlib import Path
versions, roots = {}, {}
for distribution in metadata.distributions():
    name = distribution.metadata['Name'].lower().replace('_','-').replace('.','-')
    if name in versions: raise ValueError('Duplicate installed distribution')
    versions[name] = distribution.version
    roots[name] = str(Path(distribution.locate_file('')).resolve())
origins = {}
for name in ('torch','transformers','peft','triton','safetensors','accelerate'):
    spec = util.find_spec(name)
    if spec is None or spec.origin is None: raise ValueError('Missing package origin')
    origins[name] = str(Path(spec.origin).resolve())
if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
    raise ValueError('Torch import forbidden in CPU metadata probe')
print(json.dumps({'versions':versions,'distribution_roots':roots,'package_origins':origins,
    'prefix':sys.prefix,'base_prefix':sys.base_prefix,'executable':sys.executable,
    'python_version':list(sys.version_info[:3]),'enable_user_site':site.ENABLE_USER_SITE,
    'sys_path':sys.path,'site_paths':site.getsitepackages(),'isolated':sys.flags.isolated,
    'torch_imported':False},sort_keys=True))
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def pins(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        require(match is not None and normalized(match[1]) not in result, 'Exact unique runtime pins required')
        result[normalized(match[1])] = match[2]
    require(len(result) == 76 and result['torch'] == '2.8.0+cu128'
            and result['pip'] == '24.0' and result['setuptools'] == '79.0.1', 'Fixed 76-package runtime required')
    return result


def prepare(args):
    require((platform.system(), platform.machine(), tuple(sys.version_info[:2])) ==
            ('Linux', 'x86_64', (3, 11)), 'Bootstrap must be Linux x86_64 Python 3.11')
    require(sys.flags.isolated == 1 and sys.flags.no_site == 1, 'Bootstrap must use -I -S before any shared-site imports')
    source, lock, attempt = Path(__file__).resolve(), Path(args.lock), Path(args.attempt_directory)
    require(args.expected_source_sha256 == sha(source), 'Installer source hash differs from declaration')
    require(lock.is_absolute() and lock.is_file() and not lock.is_symlink()
            and args.expected_lock_sha256 == LOCK_SHA256 and sha(lock) == LOCK_SHA256, 'Frozen runtime lock bytes required')
    require(attempt.is_absolute() and not any(v in str(attempt).lower() for v in
            ('future-host', 'placeholder', '<', '>')), 'Actual absolute attempt directory required')
    require(attempt.parent.is_dir() and not attempt.parent.is_symlink()
            and attempt.parent.stat().st_uid == os.getuid(), 'Existing per-user owned parent directory required')
    require(not attempt.exists() and not attempt.is_symlink(), 'Attempt directory exists; no retry or overwrite')
    for protected in (ROOT, Path(sys.prefix), Path(sys.base_prefix), lock):
        require(not attempt.resolve().is_relative_to(protected.resolve())
                and not protected.resolve().is_relative_to(attempt.resolve()), 'Attempt overlaps source, lock or shared environment')
    for ancestor in attempt.resolve().parents:
        require(not (ancestor/'pyvenv.cfg').exists() and not (ancestor/'conda-meta').exists(),
                'Attempt beneath an existing environment is forbidden')
    marker = attempt.with_name(attempt.name+'.consumed.json')
    require(not marker.exists() and not marker.is_symlink(), 'Attempt already consumed; no retry')
    return attempt, marker, pins(lock)


def environment(attempt):
    # An allowlist avoids forwarding unknown credential, proxy or loader vars.
    # Keep the user's actual HOME; NETRC disables implicit auth from that home.
    return {'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()), 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
            'NETRC': str(attempt/'no-netrc'), 'TMPDIR': str(attempt/'tmp'),
            'XDG_CACHE_HOME': str(attempt/'cache'), 'PIP_CONFIG_FILE': '/dev/null',
            'PYTHONNOUSERSITE': '1', 'CUDA_VISIBLE_DEVICES': ''}


def command(argv, logfile, env, deadline):
    """Every subprocess starts a new owned session; only that group is killed."""
    remaining = deadline-time.monotonic()
    require(remaining > 0, 'Overall installation deadline expired')
    with Path(logfile).open('xb') as output:
        process = None
        state = _ACTIVE_INTERRUPTION
        try:
            if state is not None:
                state['launching'] = True
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                           env=dict(env), cwd=Path(logfile).parent, start_new_session=True)
            finally:
                if state is not None:
                    state['launching'] = False
            if state is not None and state['pending'] is not None:
                state['interrupt'](state['pending'], None)
            returncode = process.wait(timeout=remaining)
        except BaseException:
            # Popen owns this child and has not reaped a still-running child.
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The owned child may have exited between poll and kill.
            if process is not None:
                process.wait(timeout=5)
            raise
    require(returncode == 0, 'Owned subprocess failed; inspect preserved log '+Path(logfile).name)


@contextmanager
def termination_handlers():
    """A termination signal enters the owned-child cleanup/failure path."""
    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {number: signal.getsignal(number) for number in signals}
    global _ACTIVE_INTERRUPTION
    previous_state = _ACTIVE_INTERRUPTION
    state = {'launching': False, 'pending': None}
    def interrupt(number, frame):
        if state['launching']:
            if state['pending'] is None:
                state['pending'] = number
            return  # deliver inside command's cleanup try after registration
        for caught in signals:
            signal.signal(caught, signal.SIG_IGN)  # permit bounded cleanup once
        error = InterruptedError('Installation interrupted by signal '+str(number))
        error.signal_number = number
        raise error
    state['interrupt'] = interrupt
    try:
        _ACTIVE_INTERRUPTION = state
        for number in signals:
            signal.signal(number, interrupt)
        yield
    finally:
        _ACTIVE_INTERRUPTION = previous_state
        for number, handler in previous.items():
            signal.signal(number, handler)


def pip_argv(python, operation, index=None, report=None, requirements=None):
    argv = [str(python), '-I', '-m', 'pip', '--isolated', '--require-virtualenv', '--disable-pip-version-check', '--no-input',
            '--keyring-provider', 'disabled', '--no-cache-dir', '--retries', '0', '--timeout', '30', operation]
    if operation == 'install':
        argv += ['--force-reinstall', '--no-deps', '--only-binary=:all:', '--index-url', index, '--report', str(report)]
        argv += ['-r', str(requirements)] if requirements else ['torch==2.8.0+cu128']
    return argv


def validate_reports(attempt, expected):
    identities = {}
    for filename, selected, hosts in (('torch-install.report.json', {'torch': expected['torch']}, {'download.pytorch.org', 'download-r2.pytorch.org'}),
            ('pypi-install.report.json', {k: v for k, v in expected.items() if k != 'torch'}, {'files.pythonhosted.org'})):
        report = json.loads((attempt/filename).read_text())
        require(report['version'] == '1' and isinstance(report['install'], list), 'Unsupported pip report schema')
        installed = {}
        for item in report['install']:
            name, version = normalized(item['metadata']['name']), item['metadata']['version']
            require(name not in installed, 'Duplicate package in installation report')
            installed[name] = version
            info = item['download_info']; url = urlsplit(info['url'])
            digest = info['archive_info']['hashes']['sha256']
            require(url.scheme == 'https' and url.hostname in hosts and url.username is None
                    and url.password is None and not url.query and url.path.endswith('.whl')
                    and re.fullmatch(r'[0-9a-f]{64}', digest) is not None, 'Public binary wheel URL and SHA256 required')
            identities[name] = {'version': version, 'url': info['url'], 'wheel_sha256': digest}
        require(installed == selected, 'Actual wheel installation set differs from frozen pins')
    return identities


def validate_metadata(attempt, expected):
    data = json.loads((attempt/'metadata.log').read_text())
    venv = (attempt/'venv').resolve()
    require(data['versions'] == expected and set(data['distribution_roots']) == set(expected)
            and set(data['package_origins']) == set(NAMES), 'Installed package inventory differs')
    require(Path(data['prefix']).resolve() == venv and Path(data['base_prefix']).resolve() != venv
            and Path(data['executable']) == attempt/'venv/bin/python'
            and data['python_version'][:2] == [3, 11] and data['enable_user_site'] is False
            and data['isolated'] == 1 and data['torch_imported'] is False, 'Python isolation proof differs')
    for location in [*data['distribution_roots'].values(), *data['package_origins'].values(), *data['site_paths']]:
        require(Path(location).is_absolute() and Path(location).resolve().is_relative_to(venv), 'Shared/inherited package location rejected')
    for location in data['sys_path']:
        if 'site-packages' in Path(location).parts or 'dist-packages' in Path(location).parts:
            require(Path(location).resolve().is_relative_to(venv), 'Inherited site path rejected')
    config = (attempt/'venv/pyvenv.cfg').read_text()
    require(re.search(r'^include-system-site-packages\s*=\s*false\s*$', config, re.M) is not None,
            'Venv system-site packages forbidden')
    return data


def boot_id():
    value = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', value) is not None,
            'Linux boot identity required')
    return value


def _install(args):
    started = time.monotonic()
    deadline = started+MAX_SECONDS-10  # reserve bounded time for owned cleanup/receipt
    attempt, marker, expected = prepare(args)
    identity = {'host': platform.node(), 'boot_id': boot_id(),
                'uid': os.getuid(), 'pid': os.getpid(), 'bootstrap_python': str(Path(sys.executable).resolve()),
                'bootstrap_python_sha256': sha(sys.executable), 'installer_sha256': sha(__file__),
                'lock_sha256': LOCK_SHA256, 'attempt_directory': str(attempt)}
    write_once(marker, {'schema_version': 1, 'status': 'installation_attempt_consumed_no_retry', **identity})
    created = False
    try:
        attempt.mkdir(mode=0o700)
        created = True
        for name in ('tmp', 'cache'):
            (attempt/name).mkdir(mode=0o700)
        env = environment(attempt)
        command([sys.executable, '-I', '-S', '-m', 'venv', str(attempt/'venv')], attempt/'venv-create.log', env, deadline)
        python = attempt/'venv/bin/python'
        derived = attempt/'pypi-75.lock'
        with derived.open('x') as handle:
            handle.write(''.join(f'{k}=={v}\n' for k, v in expected.items() if k != 'torch'))
        command(pip_argv(python, 'install', 'https://download.pytorch.org/whl/cu128',
                         attempt/'torch-install.report.json'), attempt/'torch-install.log', env, deadline)
        command(pip_argv(python, 'install', 'https://pypi.org/simple', attempt/'pypi-install.report.json', derived),
                attempt/'pypi-install.log', env, deadline)
        wheels = validate_reports(attempt, expected)
        command(pip_argv(python, 'check'), attempt/'pip-check.log', env, deadline)
        command([str(python), '-I', '-c', PROBE], attempt/'metadata.log', env, deadline)
        metadata = validate_metadata(attempt, expected)
        require(time.monotonic() < deadline, 'Overall installation deadline expired before success')
        require(sha(args.lock) == LOCK_SHA256 and sha(__file__) == args.expected_source_sha256,
                'Installer/lock bytes changed during installation')
        evidence = {p.name: sha(p) for p in attempt.iterdir() if p.is_file()}
        receipt = {'schema_version': 1, 'status': 'isolated_cpu_runtime_installation_passed', **identity,
            'elapsed_seconds': time.monotonic()-started, 'maximum_seconds': MAX_SECONDS,
            'derived_lock_sha256': sha(derived), 'evidence_sha256': evidence, 'wheel_provenance': wheels,
            'runtime_metadata': metadata, 'venv_python_sha256': sha(python),
            'venv_python_resolved': str(python.resolve()), 'venv_config_sha256': sha(attempt/'venv/pyvenv.cfg'),
            'model_calls': 0, 'GPU_actions': 0, 'torch_imports_requested': 0, 'resource_authority': False,
            'limits': ['Installation report hashes identify fetched wheels; this is not model execution or live GPU ownership.',
                       'The original bootstrap environment and source are read-only; only this new attempt was installed.']}
        write_once(attempt/'installation-receipt.json', receipt)
        return receipt
    except BaseException as error:
        failure = attempt/'installation-failure.json' if created else marker.with_name(marker.name+'.failure.json')
        write_once(failure, {'schema_version': 1,
            'status': 'isolated_cpu_runtime_installation_failed_consumed_no_retry', **identity,
            'elapsed_seconds': time.monotonic()-started, 'exception_type': type(error).__name__,
            'signal_number': getattr(error, 'signal_number', None),
            'evidence_sha256': {p.name: sha(p) for p in attempt.iterdir() if p.is_file()} if created else {},
            'model_calls': 0, 'GPU_actions': 0, 'resource_authority': False})
        raise


def install(args):
    with termination_handlers():
        return _install(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt-directory', 'lock', 'expected-lock-sha256', 'expected-source-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    receipt = install(args)
    print(json.dumps({'status': receipt['status'], 'receipt': str(Path(args.attempt_directory)/'installation-receipt.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
