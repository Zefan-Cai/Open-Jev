"""New real-Git CPU import fixtures, without host, CUDA or model callbacks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = 'scripts/frontier_v8_operational_bridge.py'
LOADER = 'scripts/frontier_v8_source_loader.py'
OPERATIONAL = (LOADER, BRIDGE, 'scripts/frontier_v8_owned_process.py',
               'scripts/frontier_v8_resource_episode.py', 'scripts/frontier_v8_live_host.py')
NATIVE = ('scripts/run_frontier_native_v8.py', 'scripts/frontier_v8_native_model.py',
          'scripts/frontier_v8_owned_process.py', 'docs/frontier-v8-native-runtime.md',
          'tests/test_run_frontier_native_v8.py', 'tests/test_frontier_v8_native_model.py',
          'tests/test_frontier_v8_owned_process.py')
PINS = {'SCIENCE_A': 'd8eeb3d1f8e8d8751476f102ab456170c277e86a',
        'NATIVE_A': '4b55c6e3f025fd92ec4396d5c61bb4dfe13cbda2',
        'PLAN_SHA': '984539a92f1aa37b93e58fea7df4511a1635e7e9f284531068bb5a0939ebc8be',
        'FREEZE_SHA': 'a710f79977ac582cdc9269368e3f442abb88e76e4bbfae89b962116ade213a5f',
        'NATIVE_DECLARATION_SHA': '53e3cea8adc09fec82aea319775569ad4d487b3b0f88cd1f094653bc70de793d'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class OperationalBridgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.roots = {role: self.root / role for role in ('scientific', 'native', 'operational')}
        for root in self.roots.values():
            root.mkdir()
            self.git(root, 'init', '-q')
        science = json.loads((ROOT / 'reports/frontier-v8-training-protocol-20261003/training-plan.json').read_bytes())
        for name in science['implementation_sha256']:
            self.write(self.roots['scientific'] / name, (ROOT / name).read_bytes())
        for name in NATIVE:
            self.write(self.roots['native'] / name, (ROOT / name).read_bytes())
        science_commit, science_files = self.commit('scientific')
        native_commit, native_files = self.commit('native')
        plan, plan_sha = self.json_file('plan.json', {'implementation_sha256': science_files})
        freeze, freeze_sha = self.json_file('freeze.json', {'source_commit': science_commit,
            'plan_sha256': plan_sha, 'implementation_sha256': science_files})
        native, native_sha = self.json_file('native.json', {'native_source_commit': native_commit,
            'native_files_sha256': native_files, 'scientific_source_commit': science_commit,
            'scientific_plan_sha256': plan_sha, 'scientific_freeze_sha256': freeze_sha})
        replacement = {'SCIENCE_A': science_commit, 'NATIVE_A': native_commit,
                       'PLAN_SHA': plan_sha, 'FREEZE_SHA': freeze_sha, 'NATIVE_DECLARATION_SHA': native_sha}
        loader = (ROOT / LOADER).read_text()
        for name, value in replacement.items():
            loader = loader.replace(PINS[name], value)
        bridge = (ROOT / BRIDGE).read_text().replace(digest((ROOT / LOADER).read_bytes()), digest(loader.encode()))
        for name in OPERATIONAL:
            raw = loader.encode() if name == LOADER else bridge.encode() if name == BRIDGE else (ROOT / name).read_bytes()
            self.write(self.roots['operational'] / name, raw)
        self.write(self.roots['operational'] / '.gitignore', b'__pycache__/\n*.pyc\n')
        operational_commit, operational_files = self.commit('operational')
        operational, operational_sha = self.json_file('operational.json', {
            'schema_version': 1, 'status': 'three_source_CPU_declaration', 'source_commit': operational_commit,
            'scientific_source_commit': science_commit, 'native_source_commit': native_commit,
            'plan_sha256': plan_sha, 'freeze_sha256': freeze_sha, 'native_declaration_sha256': native_sha,
            'files_sha256': operational_files, 'execution_available': False, 'resource_authority': False})
        self.context = {role + '_root': str(root) for role, root in self.roots.items()}
        self.context.update(plan=plan, freeze=freeze, native_declaration=native,
            operational_declaration=operational, expected_operational_commit=operational_commit,
            expected_native_declaration_sha256=native_sha,
            expected_operational_declaration_sha256=operational_sha)
        self.fixture, _ = self.json_file('context.json', self.context)

    @staticmethod
    def write(path, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    @staticmethod
    def git(root, *args):
        environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        return subprocess.check_output(['git', '-c', 'user.name=CPU Fixture', '-c',
            'user.email=fixture@example.invalid', '-C', str(root), *args],
            env=environment, stderr=subprocess.DEVNULL)

    def commit(self, role):
        root = self.roots[role]
        self.git(root, 'add', '-A')
        self.git(root, 'commit', '-q', '--no-gpg-sign', '-m', 'Separate operational CPU fixture')
        commit = self.git(root, 'rev-parse', 'HEAD').decode().strip()
        names = self.git(root, 'ls-files', '-z').decode().split('\0')[:-1]
        return commit, {name: digest((root / name).read_bytes()) for name in names}

    def json_file(self, name, value):
        raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
        path = self.root / name
        path.write_bytes(raw)
        return str(path), digest(raw)

    def child(self, body, *, environment=None, cwd=None):
        preamble = '''
            import hashlib, importlib.util, json, pathlib, sys, types
            context = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
            path = pathlib.Path(context['operational_root'])/'scripts/frontier_v8_operational_bridge.py'
            spec = importlib.util.spec_from_loader('trusted_cpu_bridge_entry', loader=None, origin=str(path))
            bridge = importlib.util.module_from_spec(spec)
            bridge.__file__ = str(path)
            exec(compile(path.read_bytes(), str(path), 'exec'), bridge.__dict__)
            calls = []
            def owned():
                calls.append('trusted_CPU_fixture_assertion')
            def snapshot():
                return (dict(sys.modules), sys.modules, sys.path, list(sys.path),
                        sys.meta_path, list(sys.meta_path), sys.path_hooks, list(sys.path_hooks),
                        sys.path_importer_cache, dict(sys.path_importer_cache), sys.dont_write_bytecode)
            def restored(before):
                assert dict(sys.modules) == before[0] and sys.modules is before[1]
                assert sys.path is before[2] and sys.path == before[3]
                assert sys.meta_path is before[4] and sys.meta_path == before[5]
                assert sys.path_hooks is before[6] and sys.path_hooks == before[7]
                assert sys.path_importer_cache is before[8] and sys.path_importer_cache == before[9]
                assert sys.dont_write_bytecode == before[10]
            def rejected(call, fragment):
                try:
                    call()
                except Exception as error:
                    assert fragment in str(error), (type(error).__name__, str(error))
                else:
                    raise AssertionError('CPU bridge unexpectedly accepted')
        '''
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c',
            textwrap.dedent(preamble) + textwrap.dedent(body), self.fixture],
            cwd=cwd or self.root, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_actual_operational_graph_origins_real_functions_and_caller_restore(self):
        self.child(r'''
            before = snapshot()
            with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                process = scope.modules['frontier_v8_owned_process']
                episode = scope.modules['frontier_v8_resource_episode']
                host = scope.modules['frontier_v8_live_host']
                assert host.process is process and host.episode is episode and episode.process is process
                assert host._canonical({'cpu': True}) == hashlib.sha256(b'{"cpu":true}').hexdigest()
                assert episode._bounded(4500, 4500) == 4500
                assert process.capture_identity.__module__ == 'scripts.frontier_v8_owned_process'
                assert host.NoBorrowPreallocationOwner.assert_owned.__module__ == 'scripts.frontier_v8_live_host'
                assert all(module.__file__.startswith(context['operational_root']+'/scripts/')
                           for module in scope.modules.values())
                assert len(scope.imported_modules) == 4 and len(calls) > 10
                assert scope.bootstrap_source_loader['path'] == context['operational_root']+'/scripts/frontier_v8_source_loader.py'
                assert scope.bootstrap_source_loader['loader'] == 'pinned_captured_source_bytes'
                assert scope.model_calls == 0 and not scope.execution_available and not scope.resource_authority
                scope.assert_current()
            restored(before)
            rejected(scope.assert_current, 'scope is closed')
        ''')
        self.assertFalse(list(self.root.rglob('*.pyc')))

    def test_legitimate_preloaded_and_forged_scripts_modules_rejected(self):
        self.child(r'''
            actual_path = pathlib.Path(context['operational_root'])/'scripts/frontier_v8_owned_process.py'
            actual_spec = importlib.util.spec_from_file_location('scripts.frontier_v8_owned_process', actual_path)
            actual = importlib.util.module_from_spec(actual_spec)
            sys.modules[actual_spec.name] = actual
            actual_spec.loader.exec_module(actual)
            before = snapshot()
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'Preloaded')
            assert actual.capture_identity.__module__ == actual_spec.name
            restored(before)
            del sys.modules[actual_spec.name]
            for name in ('scripts', 'scripts.frontier_v8_owned_process', 'jev.train', 'torch',
                         '_frontier_v8_bridge_verifier'):
                module = types.ModuleType(name)
                module.__file__ = context['operational_root']+'/scripts/frontier_v8_owned_process.py'
                sys.modules[name] = module
                before = snapshot()
                rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(),
                         'Preloaded')
                restored(before)
                del sys.modules[name]
        ''')

    def test_forged_timestamp_bytecode_is_rejected_without_executing(self):
        self.child(r'''
            import importlib._bootstrap_external as external, os
            path = pathlib.Path(context['operational_root'])/'scripts/frontier_v8_live_host.py'
            marker = path.parent.parent.parent/'forged-bytecode-ran'
            payload = compile('pathlib.Path('+repr(str(marker))+').touch()', str(path), 'exec')
            cache = pathlib.Path(importlib.util.cache_from_source(str(path)))
            cache.parent.mkdir()
            cache.write_bytes(external._code_to_timestamp_pyc(payload, int(path.stat().st_mtime), path.stat().st_size))
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(),
                     'Dirty, untracked or ignored')
            assert not marker.exists()
        ''')

    def test_foreign_meta_path_hook_and_cache_contamination_rejected(self):
        self.child(r'''
            class Hostile:
                def find_spec(self, *args):
                    raise AssertionError('hostile finder ran')
            def hook(*args):
                raise AssertionError('hostile path hook ran')
            for field, value, fragment in (('meta', Hostile(), 'meta import hook'),
                                           ('hooks', hook, 'path import hook'),
                                           ('cache', Hostile(), 'importer cache')):
                if field == 'meta': sys.meta_path.insert(0, value)
                elif field == 'hooks': sys.path_hooks.insert(0, value)
                else: sys.path_importer_cache[bridge.sysconfig.get_paths()['stdlib']] = value
                before = snapshot()
                rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), fragment)
                restored(before)
                if field == 'meta': sys.meta_path.pop(0)
                elif field == 'hooks': sys.path_hooks.pop(0)
                else: sys.path_importer_cache.pop(bridge.sysconfig.get_paths()['stdlib'])
            standard = bridge.STDLIB
            cache_before = dict(sys.path_importer_cache)
            foreign = importlib.machinery.FileFinder(context['operational_root'],
                (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES))
            sys.path_importer_cache[standard] = foreign
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'importer cache')
            sys.path_importer_cache.clear(); sys.path_importer_cache.update(cache_before)
        ''')

    def test_foreign_path_rejected_and_hostile_cwd_pythonpath_ignored(self):
        shadow = self.root / 'shadow'
        self.write(shadow / 'scripts/__init__.py', b"raise AssertionError('shadow namespace ran')\n")
        self.write(shadow / 'sitecustomize.py', b"raise AssertionError('site hook ran')\n")
        self.child(r'''
            before = snapshot()
            with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                scope.assert_current()
            restored(before)
            sys.path.insert(0, str(pathlib.Path(context['operational_root']).parent/'shadow'))
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'Foreign sys.path')
        ''', environment=dict(os.environ, PYTHONPATH=str(shadow)), cwd=shadow)

    def test_wrong_commit_and_dirty_helper_rejected(self):
        self.child(r'''
            original = context['expected_operational_commit']
            context['expected_operational_commit'] = '0'*40
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'declaration')
            context['expected_operational_commit'] = original
            target = pathlib.Path(context['operational_root'])/'scripts/frontier_v8_owned_process.py'
            target.write_bytes(target.read_bytes()+b'\n# new dirty source\n')
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'Dirty')
        ''')

    def test_bootstrap_source_loader_hash_and_bridge_origin_checked(self):
        self.child(r'''
            target = pathlib.Path(context['operational_root'])/'scripts/frontier_v8_source_loader.py'
            original = target.read_bytes()
            target.write_bytes(original+b'\n# forged bootstrap\n')
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'Pinned source loader')
            target.write_bytes(original)
            bridge.__file__ = context['operational_root']+'/scripts/elsewhere.py'
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'Bridge must run')
        ''')

    def test_scope_rejects_module_function_and_class_method_replacement(self):
        self.child(r'''
            for change in ('module', 'function', 'method', 'code', 'defaults', 'reference', 'constant'):
                before = snapshot()
                try:
                    with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                        host = scope.modules['frontier_v8_live_host']
                        process = scope.modules['frontier_v8_owned_process']
                        if change == 'module': sys.modules[host.__name__] = types.ModuleType(host.__name__)
                        elif change == 'function': host._deadline = lambda spec: None
                        elif change == 'method': host.NoBorrowPreallocationOwner.assert_owned = lambda self: None
                        elif change == 'code': process._Kernel.identity.__code__ = (lambda self, pid: None).__code__
                        elif change == 'defaults': host._origins.__defaults__ = ('forged',)
                        elif change == 'reference': host.process = types.ModuleType('forged_process')
                        elif change == 'constant': host.NODES['ms-n1-1'] = '192.0.2.1'
                except bridge.OperationalBridgeError as error:
                    assert any(word in str(error) for word in ('identity', 'constant', 'binding')), str(error)
                else: raise AssertionError(change+' accepted')
                restored(before)
        ''')

    def test_private_verifier_substitution_rejected_before_fresh_capture(self):
        self.child(r'''
            before = snapshot()
            try:
                with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                    verifier = sys.modules['_frontier_v8_bridge_verifier']
                    verifier._capture = lambda context: None
            except bridge.OperationalBridgeError as error:
                assert 'binding identity' in str(error)
            else: raise AssertionError('forged verifier accepted')
            restored(before)
        ''')

    def test_scope_container_cache_and_captured_loader_mutations_rejected(self):
        self.child(r'''
            for change in ('path', 'meta', 'hooks', 'cache', 'modules', 'loader'):
                before = snapshot()
                try:
                    with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                        if change == 'path': sys.path.append('/foreign')
                        elif change == 'meta': sys.meta_path.pop()
                        elif change == 'hooks': sys.path_hooks.clear()
                        elif change == 'cache': sys.path_importer_cache[sys.path[0]] = object()
                        elif change == 'modules': sys.modules['unverified_cpu_helper'] = types.ModuleType('unverified_cpu_helper')
                        elif change == 'loader':
                            scope.modules['frontier_v8_live_host'].__loader__.raw = b'# different bytes'
                except bridge.OperationalBridgeError as error:
                    assert any(word in str(error) for word in ('container', 'cache', 'identity', 'loader state', 'hook sequence')), str(error)
                else: raise AssertionError(change+' accepted')
                restored(before)
        ''')

    def test_sources_changed_inside_scope_and_during_import_rejected(self):
        self.child(r'''
            target = pathlib.Path(context['operational_root'])/'scripts/frontier_v8_live_host.py'
            original = target.read_bytes()
            for when in ('callback', 'during_import'):
                before = snapshot()
                count = [0]
                def assertion():
                    count[0] += 1
                    if when == 'during_import' and count[0] == 4:
                        target.write_bytes(original+b'\n# changed during import\n')
                try:
                    with bridge.load_operational_cpu_modules(context, assert_owned=assertion) as scope:
                        if when == 'callback': target.write_bytes(original+b'\n# callback changed source\n')
                except Exception as error:
                    assert 'Dirty' in str(error) or 'changed' in str(error), str(error)
                else: raise AssertionError('source mutation accepted')
                target.write_bytes(original)
                restored(before)
        ''')

    def test_ownership_loss_before_during_after_import_and_callback_restores_state(self):
        self.child(r'''
            class Lost(RuntimeError): pass
            for at in (1, 2, 3, 4, 7, 10, 12):
                before = snapshot()
                count = [0]
                entered = [False]
                def assertion():
                    count[0] += 1
                    if count[0] == at: raise Lost('CPU assertion lost')
                try:
                    with bridge.load_operational_cpu_modules(context, assert_owned=assertion):
                        entered[0] = True
                except Lost: pass
                else: raise AssertionError('loss '+str(at)+' accepted')
                assert not entered[0]
                restored(before)
            before = snapshot()
            lost = [False]
            def assertion():
                if lost[0]: raise Lost('CPU scope lost after callback')
            try:
                with bridge.load_operational_cpu_modules(context, assert_owned=assertion) as scope:
                    lost[0] = True
            except Lost: pass
            else: raise AssertionError('callback loss accepted')
            restored(before)
            rejected(scope.assert_current, 'scope is closed')
            before = snapshot()
            lost[0] = False
            try:
                with bridge.load_operational_cpu_modules(context, assert_owned=assertion):
                    lost[0] = True
                    raise RuntimeError('trusted callback failed before cleanup')
            except Lost as error:
                assert isinstance(error.__context__, RuntimeError)
            else: raise AssertionError('post-exception assertion skipped')
            restored(before)
        ''')

    def test_cpu_assertion_required_cannot_use_boolean_receipt(self):
        self.child(r'''
            for assertion in (None, True, lambda: True):
                before = snapshot()
                rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=assertion).__enter__(),
                         'callback' if not callable(assertion) else 'return None')
                restored(before)
        ''')

    def test_lazy_restoration_and_optional_model_imports_remain_undeclared(self):
        self.child(r'''
            before = snapshot()
            with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                rejected(scope.modules['frontier_v8_resource_episode']._auditor, 'Undeclared')
                rejected(lambda: __import__('torch'), 'Undeclared')
                scope.assert_current()
            restored(before)
        ''')

    def test_body_exception_import_failure_and_assertion_pollution_restore_state(self):
        self.child(r'''
            before = snapshot()
            try:
                with bridge.load_operational_cpu_modules(context, assert_owned=owned):
                    raise RuntimeError('trusted callback stopped')
            except RuntimeError as error:
                assert str(error) == 'trusted callback stopped'
            restored(before)
            def polluted():
                sys.path.append('/foreign')
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=polluted).__enter__(), 'Foreign sys.path')
            restored(before)
        ''')

    def test_cli_cpu_import_scope_and_production_refusal_before_context_read(self):
        self.child(r'''
            import contextlib, io
            # Ordinary argparse startup is outside the controlled import context.
            bridge.argparse.ArgumentParser()
            before = snapshot()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                assert bridge.main(['--import-cpu', '--context', sys.argv[1]]) == 0
            result = json.loads(output.getvalue())
            assert result['assertion_scope'] == 'trusted_CPU_assertion_fixture_not_live_ownership'
            assert result['assertion_calls'] == len(result['assertion_tags']) and result['assertion_calls'] > 10
            assert result['optional_modules_absent'] == list(bridge.OPTIONAL)
            assert result['bootstrap_source_loader']['source_sha256'] == bridge.SOURCE_LOADER_SHA256
            assert result['model_calls'] == 0 and not result['resource_authority'] and not result['execution_available']
            restored(before)
            rejected(lambda: bridge.main(['--execute', '--context', '/missing/unreadable.json']), 'Production execution unavailable')
        ''')

    def test_genuine_file_cli_imports_captured_cpu_helpers_without_execution(self):
        entry = self.roots['operational'] / BRIDGE
        result = subprocess.run([sys.executable, '-B', '-I', '-S', str(entry),
            '--import-cpu', '--context', self.fixture], cwd=self.root,
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, '')
        output = json.loads(result.stdout)
        self.assertEqual(output['source_receipt']['context'], self.context)
        self.assertEqual(len(output['imported_modules']), 4)
        self.assertEqual(output['bootstrap_source_loader']['path'], str(self.roots['operational'] / LOADER))
        self.assertEqual(output['model_calls'], 0)
        self.assertIs(output['execution_available'], False)
        self.assertIs(output['resource_authority'], False)
        self.assertEqual(output['assertion_scope'], 'trusted_CPU_assertion_fixture_not_live_ownership')
        self.assertGreater(output['assertion_calls'], 10)
        self.assertFalse(list(self.root.rglob('*.pyc')))
        refused = subprocess.run([sys.executable, '-B', '-I', '-S', str(entry),
            '--execute', '--context', '/missing/unreadable-context.json'], cwd=self.root,
            capture_output=True, text=True, timeout=15)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('Production execution unavailable', refused.stderr)
        self.assertEqual(refused.stdout, '')

    def test_exact_entry_negative_cache_retained_and_foreign_or_positive_entries_refused(self):
        self.child(r'''
            entry = str(path)
            sys.path_importer_cache[entry] = None
            before = snapshot()
            with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                scope.assert_current()
            restored(before)
            assert entry in sys.path_importer_cache and sys.path_importer_cache[entry] is None
            positive = importlib.machinery.FileFinder(entry,
                (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES))
            assert positive.path == entry
            sys.path_importer_cache[entry] = positive
            rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(),
                     'Entry-script importer cache must be negative')
            sys.path_importer_cache[entry] = None
            for foreign in (str(path.parent/'frontier_v8_source_loader.py'), str(path.parent/'foreign.py')):
                sys.path_importer_cache[foreign] = None
                rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(),
                         'Foreign path importer cache path')
                del sys.path_importer_cache[foreign]
        ''')

    def test_stdlib_org_absence_probe_never_falls_through_to_positive_import(self):
        self.child(r'''
            before = snapshot()
            with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                reached = []
                class PositiveFallback:
                    def find_spec(self, fullname, path=None, target=None):
                        reached.append(fullname)
                        raise AssertionError('org negative probe reached a positive fallback')
                hooks = list(sys.meta_path)
                sys.meta_path.insert(1, PositiveFallback())
                try:
                    try:
                        from org.python.core import PyStringMap
                    except ImportError as error:
                        assert isinstance(error, ModuleNotFoundError) and error.name == 'org'
                    else: raise AssertionError('org positive import accepted')
                finally:
                    sys.meta_path[:] = hooks
                assert reached == [] and not any(name == 'org' or name.startswith('org.') for name in sys.modules)
                rejected(lambda: __import__('torch'), 'Undeclared')
                rejected(lambda: __import__('unrelated_positive_fallback_probe'), 'Undeclared')
                scope.assert_current()
            restored(before)
        ''')

    def test_preloaded_org_package_and_submodule_aliases_are_refused(self):
        self.child(r'''
            for name in ('org', 'org.python', 'org.python.core'):
                positive = types.ModuleType(name)
                positive.__path__ = []
                positive.PyStringMap = dict
                sys.modules[name] = positive
                before = snapshot()
                rejected(lambda: bridge.load_operational_cpu_modules(context, assert_owned=owned).__enter__(), 'Preloaded')
                restored(before)
                assert sys.modules[name] is positive
                del sys.modules[name]
        ''')

    def test_org_probe_hook_instance_method_code_defaults_and_finder_binding_mutations_refused(self):
        self.child(r'''
            probe_class = bridge._DenyOrgProbe
            original = probe_class.find_spec
            code, defaults = original.__code__, original.__defaults__
            for change in ('instance', 'method', 'code', 'defaults', 'finder'):
                before = snapshot()
                try:
                    with bridge.load_operational_cpu_modules(context, assert_owned=owned) as scope:
                        probe, finder = sys.meta_path[:2]
                        if change == 'instance': probe.find_spec = lambda *args: None
                        elif change == 'method': probe_class.find_spec = lambda self, fullname, path=None, target=None: None
                        elif change == 'code': original.__code__ = (lambda self, fullname, path=None, target=None: None).__code__
                        elif change == 'defaults': original.__defaults__ = ('changed', None)
                        elif change == 'finder': finder.loaders = dict(finder.loaders)
                except bridge.OperationalBridgeError as error:
                    assert 'hook identity' in str(error) or 'finder binding' in str(error), str(error)
                else: raise AssertionError(change+' accepted')
                finally:
                    probe_class.find_spec = original
                    original.__code__, original.__defaults__ = code, defaults
                restored(before)
        ''')


if __name__ == '__main__':
    unittest.main()
