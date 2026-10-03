"""Mocked installation contracts; no pip, SSH, Torch or GPU action is run."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import install_isolated_runtime_cpu as installer


ROOT = Path(__file__).resolve().parents[1]
TORCH_SHA = '039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed'
TORCH_URL = ('https://download-r2.pytorch.org/whl/cu128/'
             'torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl#sha256='+TORCH_SHA)


def args_for(directory):
    return SimpleNamespace(attempt_directory=str(Path(directory)/'fresh'),
        lock=str(ROOT/'requirements-linux-py311-cu128-20261002.lock'),
        expected_lock_sha256=installer.LOCK_SHA256, expected_source_sha256=installer.sha(installer.__file__))


def metadata_fixture(attempt, expected):
    site = attempt/'venv/lib/python3.11/site-packages'
    return {'versions': dict(expected), 'distribution_roots': {name: str(site) for name in expected},
        'package_origins': {name: str(site/name/'__init__.py') for name in installer.NAMES},
        'prefix': str(attempt/'venv'), 'base_prefix': '/usr', 'executable': str(attempt/'venv/bin/python'),
        'python_version': [3, 11, 9], 'enable_user_site': False, 'isolated': 1, 'torch_imported': False,
        'sys_path': ['/usr/lib/python3.11', str(site)], 'site_paths': [str(site)]}


def mock_install_command(attempt, expected, calls, mutation=None):
    def run(argv, logfile, env, deadline):
        calls.append({'argv': list(argv), 'env': dict(env), 'deadline': deadline})
        Path(logfile).write_text('CPU mock command completed\n')
        if 'venv' in argv:
            (attempt/'venv/bin').mkdir(parents=True)
            (attempt/'venv/bin/python').write_bytes(b'CPU mock executable')
            (attempt/'venv/pyvenv.cfg').write_text('include-system-site-packages = false\n')
        if '--report' in argv:
            filename = Path(argv[argv.index('--report')+1])
            packages = {'torch': expected['torch']} if filename.name.startswith('torch') else {k: v for k, v in expected.items() if k != 'torch'}
            host = 'download.pytorch.org' if filename.name.startswith('torch') else 'files.pythonhosted.org'
            report = {'version': '1', 'install': [{'metadata': {'name': name, 'version': version},
                'download_info': {'url': 'https://'+host+'/'+name+'.whl',
                                  'archive_info': {'hashes': {'sha256': 'a'*64}}}} for name, version in packages.items()]}
            if filename.name == 'torch-install.report.json':
                report['install'][0]['download_info'] = {'url': TORCH_URL, 'archive_info': {'hashes': {'sha256': TORCH_SHA}}}
            # Real pip omits an already-satisfied bootstrap pip24.0 without
            # force-reinstall. Model that case rather than assuming 75 entries.
            if filename.name == 'pypi-install.report.json' and '--force-reinstall' not in argv:
                report['install'] = [v for v in report['install'] if v['metadata']['name'] != 'pip']
            if mutation and filename.name == 'pypi-install.report.json':
                mutation('report', report)
            filename.write_text(json.dumps(report))
        if Path(logfile).name == 'metadata.log':
            metadata = metadata_fixture(attempt, expected)
            if mutation:
                mutation('metadata', metadata)
            Path(logfile).write_text(json.dumps(metadata))
    return run


class IsolatedRuntimeInstallerTests(unittest.TestCase):
    def linux(self, stack):
        stack.enter_context(patch.object(installer.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(installer.platform, 'machine', return_value='x86_64'))
        stack.enter_context(patch.object(installer.sys, 'version_info', (3, 11, 9)))
        stack.enter_context(patch.object(installer.sys, 'flags', SimpleNamespace(isolated=1, no_site=1)))
        stack.enter_context(patch.object(installer, 'boot_id', return_value='01234567-89ab-cdef-0123-456789abcdef'))

    def test_environment_discards_python_pip_credentials_proxies_and_loader_overrides(self):
        inherited = {'PYTHONPATH': '/shared/inject', 'PYTHONHOME': '/shared/python', 'PIP_INDEX_URL': 'fake-private-index',
                     'PIP_CONFIG_FILE': '/shared/pip.conf', 'HTTPS_PROXY': 'fake-proxy', 'AWS_SECRET_ACCESS_KEY': 'fake-secret',
                     'HF_TOKEN': 'fake-secret', 'LD_LIBRARY_PATH': '/shared/libraries', 'CUDA_VISIBLE_DEVICES': '0,1'}
        with patch.dict(os.environ, inherited, clear=True):
            result = installer.environment(Path('/owned/new-attempt'))
        self.assertEqual(result['PIP_CONFIG_FILE'], '/dev/null')
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(result['PYTHONNOUSERSITE'], '1')
        for key in inherited:
            if key not in ('PIP_CONFIG_FILE', 'CUDA_VISIBLE_DEVICES'):
                self.assertNotIn(key, result)
        self.assertEqual(result['NETRC'], '/owned/new-attempt/no-netrc')

    def test_platform_source_lock_and_owned_parent_refusal_create_no_attempt(self):
        for bad in ('platform', 'python', 'startup', 'source', 'lock', 'owner'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                self.linux(stack)
                args = args_for(temp)
                if bad == 'platform':
                    stack.enter_context(patch.object(installer.platform, 'machine', return_value='aarch64'))
                elif bad == 'python':
                    stack.enter_context(patch.object(installer.sys, 'version_info', (3, 12, 0)))
                elif bad == 'startup':
                    stack.enter_context(patch.object(installer.sys, 'flags', SimpleNamespace(isolated=1, no_site=0)))
                elif bad == 'source':
                    args.expected_source_sha256 = '0'*64
                elif bad == 'lock':
                    args.expected_lock_sha256 = '0'*64
                elif bad == 'owner':
                    stack.enter_context(patch.object(installer.os, 'getuid', return_value=os.getuid()+1))
                with self.assertRaises(ValueError):
                    installer.prepare(args)
                self.assertFalse(Path(args.attempt_directory).exists())

    def test_source_shared_environment_and_existing_outputs_are_refused(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.linux(stack)
            args = args_for(temp)
            for protected in (ROOT, Path(temp)/'shared'):
                protected.mkdir(exist_ok=True)
                candidate = copy.copy(args)
                candidate.attempt_directory = str(protected/'fresh')
                with patch.object(installer.sys, 'prefix', str(protected)), self.assertRaises(ValueError):
                    installer.prepare(candidate)
            attempt = Path(args.attempt_directory)
            attempt.mkdir()
            with self.assertRaisesRegex(ValueError, 'no retry'):
                installer.prepare(args)
            attempt.rmdir()
            attempt.with_name(attempt.name+'.consumed.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'consumed'):
                installer.prepare(args)

    def test_timeout_kills_only_owned_new_session_and_reaps(self):
        process = Mock(pid=43210)
        process.wait.side_effect = [subprocess.TimeoutExpired(['cpu-mock'], 1), 0]
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp, patch.object(installer.subprocess, 'Popen', return_value=process) as popen, patch.object(installer.os, 'killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                installer.command(['cpu-mock'], Path(temp)/'log', installer.environment(Path(temp)), installer.time.monotonic()+1)
            kill.assert_called_once_with(43210, signal.SIGKILL)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(process.wait.call_args_list[-1].kwargs['timeout'], 5)

    def test_elapsed_deadline_prevents_starting_another_subprocess(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(installer.subprocess, 'Popen') as popen, patch.object(installer.time, 'monotonic', return_value=2000):
            with self.assertRaisesRegex(ValueError, 'deadline'):
                installer.command(['cpu-mock'], Path(temp)/'log', {}, 1000)
            popen.assert_not_called()
            self.assertFalse((Path(temp)/'log').exists())

    def test_success_records_exact_public_wheels_and_isolated_cpu_metadata(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.linux(stack)
            args = args_for(temp)
            expected = installer.pins(args.lock)
            attempt = Path(args.attempt_directory)
            calls = []
            stack.enter_context(patch.object(installer, 'command', side_effect=mock_install_command(attempt, expected, calls)))
            receipt = installer.install(args)
            self.assertIn('-S', calls[0]['argv'])
            self.assertEqual(receipt['status'], 'isolated_cpu_runtime_installation_passed')
            self.assertEqual(len(receipt['wheel_provenance']), 76)
            self.assertEqual(receipt['wheel_provenance']['torch']['url'], TORCH_URL)
            self.assertEqual(receipt['wheel_provenance']['torch']['wheel_sha256'], TORCH_SHA)
            self.assertEqual(receipt['runtime_metadata']['versions'], expected)
            self.assertEqual(receipt['GPU_actions'], 0)
            self.assertFalse(receipt['resource_authority'])
            self.assertTrue((attempt/'installation-receipt.json').is_file())
            pip_calls = [call for call in calls if 'pip' in call['argv']]
            self.assertEqual(len(pip_calls), 3)
            for call in pip_calls:
                self.assertEqual(call['env']['PIP_CONFIG_FILE'], '/dev/null')
                self.assertEqual(call['env']['CUDA_VISIBLE_DEVICES'], '')
                for flag in ('-I', '--isolated', '--no-cache-dir', '--retries', '--no-input'):
                    self.assertIn(flag, call['argv'])
                self.assertEqual(call['argv'][call['argv'].index('--retries')+1], '0')
            installs = [call['argv'] for call in pip_calls if 'install' in call['argv']]
            self.assertEqual(installs[0][installs[0].index('--index-url')+1], 'https://download.pytorch.org/whl/cu128')
            self.assertEqual(installs[1][installs[1].index('--index-url')+1], 'https://pypi.org/simple')
            for argv in installs:
                self.assertIn('--only-binary=:all:', argv)
                self.assertIn('--no-deps', argv)
                self.assertIn('--force-reinstall', argv)
            self.assertEqual(len((attempt/'pypi-75.lock').read_text().splitlines()), 75)
            self.assertNotIn('torch==', (attempt/'pypi-75.lock').read_text())
            self.assertFalse(any('nvidia-smi' in call['argv'] for call in calls))
            with self.assertRaises(ValueError):
                installer.install(args)

    def test_failed_version_origin_site_or_wheel_proof_preserves_consumed_failure(self):
        for bad in ('version', 'extra', 'origin', 'site', 'user_site', 'url', 'hash'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                self.linux(stack)
                args = args_for(temp)
                attempt = Path(args.attempt_directory)
                def mutation(phase, data):
                    if phase == 'metadata':
                        if bad == 'version': data['versions']['torch'] = 'different'
                        elif bad == 'extra': data['versions']['wheel'] = '0.45'
                        elif bad == 'origin': data['package_origins']['torch'] = '/shared/torch/__init__.py'
                        elif bad == 'site': data['sys_path'].append('/shared/site-packages')
                        elif bad == 'user_site': data['enable_user_site'] = True
                    elif phase == 'report':
                        if bad == 'url': data['install'][0]['download_info']['url'] = 'https://user:fake@files.pythonhosted.org/a.whl'
                        elif bad == 'hash': data['install'][0]['download_info']['archive_info']['hashes']['sha256'] = 'bad'
                calls = []
                stack.enter_context(patch.object(installer, 'command', side_effect=mock_install_command(attempt, installer.pins(args.lock), calls, mutation)))
                with self.assertRaises(ValueError):
                    installer.install(args)
                failure = json.loads((attempt/'installation-failure.json').read_text())
                self.assertEqual(failure['status'], 'isolated_cpu_runtime_installation_failed_consumed_no_retry')
                self.assertTrue(failure['evidence_sha256'])
                self.assertFalse((attempt/'installation-receipt.json').exists())
                with self.assertRaises(ValueError):
                    installer.install(args)

    def test_overall_budget_expiry_has_failure_receipt_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.linux(stack)
            args = args_for(temp)
            attempt = Path(args.attempt_directory)
            calls = []
            stack.enter_context(patch.object(installer, 'command', side_effect=mock_install_command(attempt, installer.pins(args.lock), calls)))
            stack.enter_context(patch.object(installer.time, 'monotonic', side_effect=[0, 1791, 1791]))
            with self.assertRaisesRegex(ValueError, 'deadline'):
                installer.install(args)
            failure = json.loads((attempt/'installation-failure.json').read_text())
            self.assertEqual(failure['elapsed_seconds'], 1791)
            self.assertEqual(len(calls), 5)
            self.assertFalse((attempt/'installation-receipt.json').exists())

    def test_other_existing_env_ancestors_are_refused(self):
        for marker in ('pyvenv.cfg', 'conda-meta'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                self.linux(stack)
                root = Path(temp)
                if marker == 'pyvenv.cfg':
                    (root/marker).write_text('include-system-site-packages = false\n')
                else:
                    (root/marker).mkdir()
                (root/'user').mkdir()
                args = args_for(root/'user')
                with self.assertRaisesRegex(ValueError, 'existing environment'):
                    installer.prepare(args)
                self.assertFalse(Path(args.attempt_directory).exists())

    def test_termination_signals_clean_owned_child_write_failure_and_restore_handlers(self):
        for target in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with self.subTest(signal=target), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                self.linux(stack)
                args = args_for(temp)
                handlers, original = {}, object()
                def set_signal(number, handler):
                    handlers[number] = handler
                stack.enter_context(patch.object(installer.signal, 'getsignal', return_value=original))
                changed = stack.enter_context(patch.object(installer.signal, 'signal', side_effect=set_signal))
                process = Mock(pid=45678)
                process.poll.return_value = None
                waits = []
                def wait(timeout):
                    waits.append(timeout)
                    if len(waits) == 1:
                        handlers[target](target, None)
                    return 0
                process.wait.side_effect = wait
                stack.enter_context(patch.object(installer.subprocess, 'Popen', return_value=process))
                killed = stack.enter_context(patch.object(installer.os, 'killpg'))
                with self.assertRaises(InterruptedError):
                    installer.install(args)
                killed.assert_called_once_with(45678, signal.SIGKILL)
                self.assertEqual(waits[-1], 5)
                failure = json.loads((Path(args.attempt_directory)/'installation-failure.json').read_text())
                self.assertEqual(failure['signal_number'], target)
                self.assertEqual(failure['exception_type'], 'InterruptedError')
                self.assertIs(handlers[signal.SIGTERM], original)
                self.assertIs(handlers[signal.SIGHUP], original)
                self.assertIs(handlers[signal.SIGINT], original)
                self.assertFalse((Path(args.attempt_directory)/'installation-receipt.json').exists())

    def test_signal_during_popen_return_is_deferred_until_child_registration(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.linux(stack)
            args = args_for(temp)
            handlers = {}
            stack.enter_context(patch.object(installer.signal, 'getsignal', return_value=signal.SIG_DFL))
            stack.enter_context(patch.object(installer.signal, 'signal', side_effect=lambda n, h: handlers.update({n: h})))
            process = Mock(pid=56789)
            process.poll.return_value = None
            process.wait.return_value = 0
            def popen(*arguments, **options):
                handlers[signal.SIGTERM](signal.SIGTERM, None)
                self.assertTrue(installer._ACTIVE_INTERRUPTION['launching'])
                return process
            stack.enter_context(patch.object(installer.subprocess, 'Popen', side_effect=popen))
            killed = stack.enter_context(patch.object(installer.os, 'killpg'))
            with self.assertRaises(InterruptedError):
                installer.install(args)
            killed.assert_called_once_with(56789, signal.SIGKILL)
            process.wait.assert_called_once_with(timeout=5)
            failure = json.loads((Path(args.attempt_directory)/'installation-failure.json').read_text())
            self.assertEqual(failure['signal_number'], signal.SIGTERM)
            self.assertIsNone(installer._ACTIVE_INTERRUPTION)

    def test_exact_official_torch_hosts_accept_real_index_url_and_reject_suffixes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = installer.pins(ROOT/'requirements-linux-py311-cu128-20261002.lock')
            run = mock_install_command(root, expected, [])
            for name in ('torch-install.report.json', 'pypi-install.report.json'):
                run(['pip', 'install', '--force-reinstall', '--report', str(root/name)], root/(name+'.log'), {}, 100)
            report = json.loads((root/'torch-install.report.json').read_text())
            for host, accepted in (('download.pytorch.org', True), ('download-r2.pytorch.org', True),
                                   ('download-r2.pytorch.org.unrelated.test', False),
                                   ('unrelated-download-r2.pytorch.org', False), ('files.pythonhosted.org', False)):
                with self.subTest(host=host):
                    report['install'][0]['download_info']['url'] = TORCH_URL.replace('download-r2.pytorch.org', host)
                    (root/'torch-install.report.json').write_text(json.dumps(report))
                    if accepted:
                        self.assertEqual(len(installer.validate_reports(root, expected)), 76)
                    else:
                        with self.assertRaisesRegex(ValueError, 'Public binary wheel'):
                            installer.validate_reports(root, expected)


if __name__ == '__main__':
    unittest.main()
