"""Captured-byte operational CPU imports in a trusted clean interpreter.

This is source provenance and a scoped CPU assertion callback, not a Python
sandbox, live ownership check, GPU lease, model bridge or recovery protocol.
Raw Python references retained by a trusted callback cannot be revoked.
"""
import argparse
import _imp
from contextlib import contextmanager
import hashlib
import importlib
from importlib import machinery, util
import json
import os
from pathlib import Path
import stat
import sys
import sysconfig
import threading
from types import FunctionType, ModuleType
import zipimport


SOURCE_LOADER_SHA256 = '6f33f86c9b37bd69612e7d0e79772b23c27540cae7a4191526fc47dbf75e2ede'
HELPERS = ('frontier_v8_owned_process', 'frontier_v8_resource_episode', 'frontier_v8_live_host')
OPTIONAL = ('torch', 'transformers', 'peft', 'triton', 'safetensors', 'accelerate')
STDLIB = sysconfig.get_paths()['stdlib']


class OperationalBridgeError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise OperationalBridgeError(message)


def _read(path):
    _require(path.is_absolute() and '..' not in path.parts, 'Absolute helper path required')
    for part in (path, *path.parents):
        _require(not stat.S_ISLNK(part.lstat().st_mode), 'Symlink helper path rejected')
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), 'Regular helper source required')
    identity = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                             info.st_mtime_ns, info.st_ctime_ns)
    with path.open('rb') as handle:
        _require(identity(os.fstat(handle.fileno())) == identity(before), 'Helper replaced before read')
        raw = handle.read()
        _require(identity(os.fstat(handle.fileno())) == identity(before), 'Helper changed during read')
    _require(identity(path.lstat()) == identity(before), 'Helper path changed during read')
    return raw


def _clean_interpreter():
    _require(threading.current_thread() is threading.main_thread(), 'Main-thread import scope required')
    _require(all(hook in (machinery.BuiltinImporter, machinery.FrozenImporter, machinery.PathFinder)
                 for hook in sys.meta_path), 'Foreign meta import hook rejected')
    _require(not any(name == 'jev' or name.startswith('jev.') or name == 'scripts'
                     or name.startswith('scripts.') or name.startswith('_frontier_v8_bridge_')
                     for name in sys.modules), 'Preloaded source namespace rejected')
    _require(not any(name == prefix or name.startswith(prefix + '.')
                     for name in sys.modules for prefix in OPTIONAL), 'Preloaded optional module rejected')
    stdlib = STDLIB
    paths = {stdlib, str(Path(stdlib) / 'lib-dynload'),
             str(Path(stdlib).parent / ('python' + str(sys.version_info.major) +
                                      str(sys.version_info.minor) + '.zip'))}
    _require(all(type(path) is str and path in paths for path in sys.path), 'Foreign sys.path rejected')
    expected = machinery.FileFinder.path_hook()
    for hook in sys.path_hooks:
        if hook is zipimport.zipimporter:
            continue
        _require(isinstance(hook, FunctionType) and hook.__code__ is expected.__code__
                 and hook.__closure__ is not None and len(hook.__closure__) == 2
                 and hook.__closure__[0].cell_contents is machinery.FileFinder,
                 'Foreign path import hook rejected')
        details = hook.__closure__[1].cell_contents
        allowed = {machinery.SourceFileLoader: machinery.SOURCE_SUFFIXES,
                   machinery.SourcelessFileLoader: machinery.BYTECODE_SUFFIXES,
                   machinery.ExtensionFileLoader: machinery.EXTENSION_SUFFIXES}
        _require(all(loader in allowed and suffixes == allowed[loader]
                     for loader, suffixes in details), 'Foreign path import hook loader rejected')
    _clean_cache(sys.path_importer_cache, paths)
    return stdlib


def _clean_cache(cache, paths):
    allowed = {machinery.SourceFileLoader: machinery.SOURCE_SUFFIXES,
               machinery.SourcelessFileLoader: machinery.BYTECODE_SUFFIXES,
               machinery.ExtensionFileLoader: machinery.EXTENSION_SUFFIXES}
    for path, finder in cache.items():
        _require(type(path) is str and (path in paths or
                 Path(path).is_relative_to(Path(STDLIB))),
                 'Foreign path importer cache path rejected')
        if finder is None:
            continue
        if type(finder) is zipimport.zipimporter:
            _require(finder.archive == path and not finder.prefix, 'Foreign zip importer cache rejected')
            continue
        _require(type(finder) is machinery.FileFinder and finder.path == path
                 and set(finder.__dict__) == {
                 'path', '_loaders', '_path_mtime', '_path_cache', '_relaxed_path_cache'}
                 and all(loader in allowed and suffix in allowed[loader]
                         for suffix, loader in finder._loaders), 'Foreign path importer cache rejected')


def _function_state(function):
    return (id(function.__code__), id(function.__defaults__), _basic(function.__defaults__),
            id(function.__kwdefaults__), _basic(function.__kwdefaults__),
            id(function.__closure__), _basic(tuple(cell.cell_contents
                for cell in function.__closure__)) if function.__closure__ else None)


def _basic(value):
    # Detect in-place edits to source constants without traversing live objects.
    if type(value) in (str, bytes, int, float, bool, type(None)):
        return value
    if type(value) in (tuple, list, set, frozenset):
        return (type(value), tuple(_basic(item) for item in value))
    if type(value) is dict:
        return (dict, tuple((key, _basic(item)) for key, item in value.items()))
    return id(value)


class _Bindings:
    def __init__(self, module):
        self.module, self.attributes = module, dict(module.__dict__)
        self.values = {name: _basic(value) for name, value in self.attributes.items()
                       if name != '__builtins__'}
        self.functions, self.classes = {}, {}
        for value in self.attributes.values():
            if isinstance(value, FunctionType):
                self.functions[value] = _function_state(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                attributes = dict(value.__dict__)
                self.classes[value] = attributes
                for member in attributes.values():
                    functions = ((member.__func__,) if isinstance(member, (staticmethod, classmethod))
                        else (member.fget, member.fset, member.fdel) if isinstance(member, property)
                        else (member,))
                    for function in functions:
                        if isinstance(function, FunctionType):
                            self.functions[function] = _function_state(function)

    def audit(self):
        _require(set(self.module.__dict__) == set(self.attributes)
                 and all(self.module.__dict__[name] is value for name, value in self.attributes.items()),
                 'Loaded module binding identity differs: ' + self.module.__name__)
        _require(all(_basic(self.module.__dict__[name]) == value for name, value in self.values.items()),
                 'Loaded source constant changed: ' + self.module.__name__)
        _require(all(_function_state(function) == state for function, state in self.functions.items()),
                 'Loaded function/code/default identity differs: ' + self.module.__name__)
        for cls, attributes in self.classes.items():
            _require(set(cls.__dict__) == set(attributes)
                     and all(cls.__dict__[name] is value for name, value in attributes.items()),
                     'Loaded class descriptor identity differs: ' + self.module.__name__)


class CPUImportScope:
    """Inspection handles close with the scope; raw trusted references do not."""
    def __init__(self, modules, receipt, audit, bootstrap):
        self.modules, self.source_receipt, self.imported_modules = modules, receipt, audit
        self.bootstrap_source_loader = bootstrap
        self.execution_available = self.resource_authority = False
        self.model_calls = 0
        self._check, self._closed = None, False

    def assert_current(self):
        _require(not self._closed, 'CPU import scope is closed')
        self._check()


@contextmanager
def load_operational_cpu_modules(context, *, assert_owned):
    """Import only process/parser/preallocation helpers; invoke no host functions.

    ``assert_owned`` is a trusted supplied CPU assertion, never resource authority.
    It brackets capture/import/audit and the trusted caller's CPU inspection scope.
    """
    _require(callable(assert_owned), 'Explicit trusted CPU assertion callback required')
    stdlib = _clean_interpreter()
    original = (sys.modules, sys.path, sys.meta_path, sys.path_hooks, sys.path_importer_cache)
    before = (dict(sys.modules), list(sys.path), list(sys.meta_path), list(sys.path_hooks),
              dict(sys.path_importer_cache))
    parents = {name: dict(module.__dict__) for name, module in before[0].items()
               if isinstance(module, ModuleType) and '__path__' in module.__dict__}
    bytecode, scope = sys.dont_write_bytecode, None
    _imp.acquire_lock()
    try:
        def owned():
            _require(assert_owned() is None, 'CPU assertion must raise on loss and return None')

        owned()
        _clean_interpreter()
        sys.path = [stdlib, str(Path(stdlib) / 'lib-dynload')]
        sys.meta_path = [machinery.BuiltinImporter, machinery.FrozenImporter, machinery.PathFinder]
        sys.path_hooks = [zipimport.zipimporter, machinery.FileFinder.path_hook(
            (machinery.SourceFileLoader, machinery.SOURCE_SUFFIXES),
            (machinery.ExtensionFileLoader, machinery.EXTENSION_SUFFIXES))]
        sys.path_importer_cache = {}
        sys.dont_write_bytecode = True
        path = Path(__file__).parent / 'frontier_v8_source_loader.py'
        raw = _read(path)
        _require(hashlib.sha256(raw).hexdigest() == SOURCE_LOADER_SHA256, 'Pinned source loader bytes differ')
        name = '_frontier_v8_bridge_verifier'
        spec = util.spec_from_loader(name, loader=None, origin=str(path))
        verifier = util.module_from_spec(spec)
        verifier.__file__ = str(path)
        sys.modules[name] = verifier
        exec(compile(raw, str(path), 'exec', dont_inherit=True), verifier.__dict__)
        binding = _Bindings(verifier)
        owned(); binding.audit()
        receipt, sources, identities = verifier._verified(context)
        owned(); binding.audit()
        bridge_path = Path(receipt['operational']['root']) / 'scripts/frontier_v8_operational_bridge.py'
        _require(Path(__file__) == bridge_path and sources['operational'].get(
            'scripts/frontier_v8_operational_bridge.py') == _read(bridge_path),
            'Bridge must run from its declared source bytes')
        loaders = {'scripts': verifier._CapturedLoader(None, b'', True)}
        for helper in HELPERS:
            relative = 'scripts/' + helper + '.py'
            _require(relative in sources['operational'], 'Operational helper missing from declared closure')
            loaders['scripts.' + helper] = verifier._CapturedLoader(
                str(Path(receipt['operational']['root']) / relative), sources['operational'][relative])
        finder = verifier._Finder(loaders)
        sys.meta_path.insert(0, finder)
        modules, bindings = {}, [binding]
        for helper in HELPERS:
            owned()
            for current in bindings:
                current.audit()
            module = importlib.import_module('scripts.' + helper)
            modules[helper] = module
            bindings.append(_Bindings(module))
            owned()
            for current in bindings:
                current.audit()
        bindings.append(_Bindings(sys.modules['scripts']))
        import_state = (sys.modules, sys.path, list(sys.path), sys.meta_path, list(sys.meta_path),
                        sys.path_hooks, list(sys.path_hooks), sys.path_importer_cache)
        imported = dict(sys.modules)
        cache = dict(sys.path_importer_cache)
        provenance = verifier._audit(loaders)
        loader_state = {name: _basic(loader.__dict__) for name, loader in loaders.items()}

        def audit_import_state():
            _require(sys.modules is import_state[0] and sys.path is import_state[1]
                     and sys.path == import_state[2] and sys.meta_path is import_state[3]
                     and sys.meta_path == import_state[4] and sys.path_hooks is import_state[5]
                     and sys.path_hooks == import_state[6] and sys.path_importer_cache is import_state[7]
                     and sys.dont_write_bytecode is True, 'Scoped import container changed')
            _require(all(sys.modules.get(name) is module for name, module in imported.items())
                     and all(name in imported or name.split('.')[0] in sys.stdlib_module_names
                             for name in sys.modules), 'Scoped loaded module identity changed')
            _require(all(sys.path_importer_cache.get(path) is value for path, value in cache.items()),
                     'Scoped path importer cache changed')
            _clean_cache(sys.path_importer_cache, set(import_state[2]))
            _require(all(_basic(loader.__dict__) == loader_state[name] for name, loader in loaders.items()),
                     'Captured loader state changed')

        def audit():
            owned()
            for current in bindings:
                current.audit()
            audit_import_state()
            result = verifier._audit(loaders)
            _require(result == provenance, 'Captured import provenance changed')
            owned(); binding.audit()
            _require(verifier._capture(context) == (receipt, sources, identities),
                     'Sources changed inside operational CPU scope')
            owned()
            for current in bindings:
                current.audit()
            audit_import_state()
            return result

        audit_result = audit()
        scope = CPUImportScope(modules, receipt, audit_result, {
            'path': str(path), 'source_sha256': SOURCE_LOADER_SHA256,
            'loader': 'pinned_captured_source_bytes', 'private_module_alias': name,
            'source_role': 'operational'})
        scope._check = audit
        try:
            yield scope
        finally:
            audit()
    finally:
        if scope is not None:
            scope._closed = True
        sys.modules, sys.path, sys.meta_path, sys.path_hooks, sys.path_importer_cache = original
        for name in set(sys.modules) - set(before[0]):
            del sys.modules[name]
        sys.modules.update(before[0])
        for name, attributes in parents.items():
            before[0][name].__dict__.clear()
            before[0][name].__dict__.update(attributes)
        sys.path[:] = before[1]
        sys.meta_path[:] = before[2]
        sys.path_hooks[:] = before[3]
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(before[4])
        sys.dont_write_bytecode = bytecode
        _imp.release_lock()


def execute(*args, **kwargs):
    raise OperationalBridgeError('Production execution unavailable; CPU imports grant no model or GPU authority')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--import-cpu', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--context')
    args = parser.parse_args(argv)
    if args.execute:
        execute()
    context_path = Path(args.context)
    raw = _read(context_path)
    context = json.loads(raw)
    _require(all(not context_path.is_relative_to(Path(context[role + '_root']))
                 for role in ('scientific', 'native', 'operational')), 'Context overlaps a source checkout')
    assertions = []
    def cpu_assertion_fixture():
        assertions.append('trusted_CPU_assertion_fixture_not_live_ownership')
    with load_operational_cpu_modules(context, assert_owned=cpu_assertion_fixture) as scope:
        scope.assert_current()
        result = {'source_receipt': scope.source_receipt, 'imported_modules': scope.imported_modules,
                  'bootstrap_source_loader': scope.bootstrap_source_loader,
                  'scope': 'captured_operational_CPU_import_inspection_only',
                  'assertion_scope': 'trusted_CPU_assertion_fixture_not_live_ownership',
                  'optional_modules_absent': list(OPTIONAL),
                  'execution_available': False, 'resource_authority': False, 'model_calls': 0}
    _require(_read(context_path) == raw, 'Context bytes changed during CPU imports')
    result['assertion_calls'] = len(assertions)
    result['assertion_tags'] = assertions
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
