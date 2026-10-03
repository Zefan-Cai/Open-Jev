"""Hand-authored packing fixtures; never read real reserved dataset bodies."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from jev.data import validate_records
from scripts import prepare_frontier_training_v8 as preparation


def row(index, group, source, kind='choice'):
    return {'id': 'fixture-'+str(index), 'group_id': group, 'split': 'train', 'source': source,
        'state': {'account': 'fixture-account-'+str(index), 'status': 'active'},
        'question': 'Which status follows?' if kind == 'choice' else 'Is active established?',
        'kind': kind, 'options': ['active', 'closed'] if kind == 'choice' else ['no', 'yes'],
        'target': [1.0, 0.0] if kind == 'choice' else [0.0, 1.0],
        'metadata': {'family': 'policy', 'template_id': 'fixture-template-'+group,
            'entity_ids': [group], 'provenance': {'type': 'synthetic', 'license': 'CC0-1.0',
                'split_policy': 'whole-parent fixtures', 'generator_version': source,
                'seed': 1, 'group_index': index, 'variant': index}}}


def summary(rows):
    value = validate_records(rows)
    return {key: value[key] for key in preparation.SUMMARY_KEYS}


class PackingFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The real declared counts are exercised using only invented fixture rows.
        excluded = []
        for group in range(335):
            excluded.extend(['excluded-'+str(group)]*(6 if group < 177 else 5))
        kept = [group for i in range(485) for group in ['kept-'+str(i)]*4]
        cls.old = [row(i, group, 'boundary-controls-v7', 'choice' if i < 1133 else 'noul')
                   for i, group in enumerate(excluded)]
        cls.old += [row(1852+i, group, 'boundary-controls-v7', 'choice' if i < 1455 else 'noul')
                    for i, group in enumerate(kept)]
        new_groups = [group for i in range(14) for group in ['new-'+str(i)]*(71 if i < 12 else 70)]
        cls.new = [row(3792+i, group, 'frontier-controls-v8', 'choice' if i < 496 else 'noul')
                   for i, group in enumerate(new_groups)]
        cls.selected = cls.old[1852:]+cls.new

    def fixture(self, directory, *, blocked=False):
        base = Path(directory)
        previous, frontier = base/'previous', base/'frontier'
        previous.mkdir(); frontier.mkdir()
        old_raw = b''.join(('  '+json.dumps(r)+' \n').encode() for r in self.old)
        new_raw = b''.join((json.dumps(r)+'\n').encode() for r in self.new)
        (previous/'train.jsonl').write_bytes(old_raw)
        (frontier/'train.jsonl').write_bytes(new_raw)
        for source in (previous, frontier):
            (source/'manifest.json').write_bytes(b'{"fixture": "manifest"}\n')
        (previous/'comparison-plan.json').write_bytes(b'{"fixture": "historical metadata only"}\n')
        copies = {}
        for split in preparation.RESERVED:
            name = split+'.jsonl'
            raw = ('RESERVED BYTE COPY ONLY '+split+'\n').encode()
            (frontier/name).write_bytes(raw); copies[name] = preparation.sha(raw)
        retained = {}
        for origin in ('v4', 'v5', 'v6', 'v7'):
            for split in ('test', 'ood'):
                name = f'observed-regression/{origin}/{split}.jsonl'
                raw = ('OBSERVED BYTE COPY ONLY '+origin+' '+split+'\n').encode()
                path = previous/(split+'.jsonl') if origin == 'v7' else previous/name
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
                retained[name] = preparation.sha(raw)
        bindings = {'previous_manifest_sha256': preparation.sha((previous/'manifest.json').read_bytes()),
            'previous_plan_sha256': preparation.sha((previous/'comparison-plan.json').read_bytes()),
            'frontier_manifest_sha256': preparation.sha((frontier/'manifest.json').read_bytes()),
            'frontier_freeze_receipt_sha256': 'f'*64,
            'train_files_sha256': {'prepared_v7': preparation.sha(old_raw), 'frontier_v8': preparation.sha(new_raw)},
            'new_reserved_files_sha256': copies, 'retained_regression_files_sha256': retained,
            'full_observed_inventory': {'fixture': {'rows': 1128}}}
        lock = {'input_bindings': bindings, 'expected_summary': summary(self.old+self.new),
                'source_counts': {'prepared_v7': dict(Counter(r['source'] for r in self.old)),
                                  'frontier_v8': dict(Counter(r['source'] for r in self.new))}}
        auditor = base/'auditor.py'; auditor.write_bytes(b'# independently authored fixture auditor\n')
        declaration = {'schema_version': 1, 'status': 'independent_whole_parent_exclusion_declared',
            'audit_source_sha256': preparation.sha(auditor.read_bytes()), 'input_bindings': bindings,
            'blocked_full_preflight_sha256': 'b'*64, 'group_hash_algorithm': 'sha256_canonical_json',
            'excluded_old_train_group_sha256': sorted({preparation.json_sha(r['group_id']) for r in self.old[:1852]}),
            'excluded_old_train_rows': 1852, 'train_summary': summary(self.selected)}
        declaration_path = base/'exclusion.json'
        declaration_path.write_text(json.dumps(declaration))
        receipt = {'schema_version': 1,
            'status': 'independent_frontier_training_inputs_blocked' if blocked else 'independent_frontier_training_inputs_passed',
            'model_calls': 0, 'failed_checks': ['inherited overlap'] if blocked else [],
            'audit_source_sha256': preparation.sha(auditor.read_bytes()),
            'input_bindings': bindings if blocked else {**bindings, 'exclusion_declaration_sha256': preparation.sha(declaration_path.read_bytes())},
            'train_summary': lock['expected_summary'] if blocked else declaration['train_summary']}
        receipt_path = base/'preflight.json'; receipt_path.write_text(json.dumps(receipt))
        return previous, frontier, lock, auditor, receipt_path, declaration_path

    def packed(self, paths, output, *, exclusion=True):
        previous, frontier, lock, auditor, receipt, declaration = paths
        with patch.object(preparation, 'load_lock', return_value=lock), \
             patch.object(preparation, 'committed_source', return_value={'commit': 'c'*40, 'files_sha256': {}}), \
             patch.object(preparation, 'AUDITOR', auditor):
            return preparation.prepare(previous, frontier, output, receipt, declaration if exclusion else None)

    def test_declared_derivative_preserves_raw_lines_order_metadata_and_reserved_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory); output = Path(directory)/'output'
            manifest = self.packed(paths, output)
            expected = b''.join(paths[0].joinpath('train.jsonl').read_bytes().splitlines(keepends=True)[1852:])
            expected += paths[1].joinpath('train.jsonl').read_bytes()
            self.assertEqual((output/'train.jsonl').read_bytes(), expected)
            self.assertEqual([json.loads(line) for line in expected.splitlines()], self.selected)
            self.assertEqual(manifest['summary'], summary(self.selected))
            self.assertEqual(manifest['train_membership']['ordered_rows_sha256'], preparation.json_sha(self.selected))
            self.assertEqual(manifest['whole_parent_exclusion']['excluded_old_groups'], 335)
            for name, digest in manifest['files_sha256'].items():
                self.assertEqual(preparation.sha((output/name).read_bytes()), digest)
            for name, digest in manifest['retained_files_sha256'].items():
                self.assertEqual(preparation.sha((output/name).read_bytes()), digest)
            self.assertEqual(manifest['selected_observed_total'], 904)
            self.assertEqual(manifest['full_observed_screen_total'], 1128)
            self.assertFalse(manifest['training_mixture_frozen'])
            self.assertFalse(manifest['initializer_frozen'])
            self.assertNotIn('training_argv', manifest)

    def test_blocked_full_variant_never_reads_reserved_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory, blocked=True); output = Path(directory)/'output'
            with patch.object(preparation, 'bound_bytes', wraps=preparation.bound_bytes) as reads:
                with self.assertRaisesRegex(ValueError, 'did not pass'):
                    self.packed(paths, output, exclusion=False)
            self.assertFalse(output.exists())
            self.assertFalse(any(Path(call.args[0]).name in {s+'.jsonl' for s in preparation.RESERVED}
                                 for call in reads.call_args_list))

    def test_passed_derivative_receipt_cannot_silently_filter_full_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory); output = Path(directory)/'output'
            with self.assertRaisesRegex(ValueError, 'not bound'):
                self.packed(paths, output, exclusion=False)
            self.assertFalse(output.exists())

    def test_renamed_observed_row_cannot_replace_locked_train(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory); output = Path(directory)/'output'
            injected = copy.deepcopy(self.old[0])
            injected.update(id='renamed-observed-row', group_id='renamed-observed-parent', split='train')
            # Renaming and mislabelling provenance cannot change the public Train lock.
            paths[0].joinpath('train.jsonl').write_bytes((json.dumps(injected)+'\n').encode())
            with self.assertRaisesRegex(ValueError, 'hash differs before parsing'):
                self.packed(paths, output)
            self.assertFalse(output.exists())

    def test_forged_manifest_and_wrong_new_reserved_hash_stop_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory); output = Path(directory)/'output'
            paths[0].joinpath('manifest.json').write_bytes(b'{"forged": true}\n')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                self.packed(paths, output)
            self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory); output = Path(directory)/'output'
            paths[1].joinpath('test.jsonl').write_bytes(b'changed reserved bytes\n')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                self.packed(paths, output)
            self.assertFalse(output.exists())

    def test_exclusion_cannot_duplicate_or_name_a_new_parent(self):
        for mode in ('duplicate', 'new'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                paths = self.fixture(directory); output = Path(directory)/'output'
                declaration = json.loads(paths[-1].read_text())
                declaration['excluded_old_train_group_sha256'][-1] = (
                    declaration['excluded_old_train_group_sha256'][0] if mode == 'duplicate'
                    else preparation.json_sha(self.new[0]['group_id']))
                paths[-1].write_text(json.dumps(declaration))
                with self.assertRaisesRegex(ValueError, 'exclusion hashes|only original old'):
                    self.packed(paths, output)
                self.assertFalse(output.exists())

    def test_old_pass_receipt_cannot_authorize_different_parent_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory); output = Path(directory)/'output'
            declaration = json.loads(paths[-1].read_text())
            declaration['explanation'] = 'A separately changed declaration needs its own preflight binding.'
            paths[-1].write_text(json.dumps(declaration))
            with self.assertRaisesRegex(ValueError, 'not bound'):
                self.packed(paths, output)
            self.assertFalse(output.exists())

    def test_misassigned_split_duplicate_id_parent_and_visible_input_are_rejected(self):
        old = row(1, 'old', 'boundary-controls-v7'); new = row(2, 'new', 'frontier-controls-v8')
        lock = {'source_counts': {'prepared_v7': {old['source']: 1}, 'frontier_v8': {new['source']: 1}},
                'expected_summary': summary([old, new])}
        for mode, pattern in [('split', 'misassigned'), ('id', 'duplicate id'), ('parent', 'Parent shared'),
                              ('visible', 'Repeated model-visible')]:
            changed = copy.deepcopy(new)
            if mode == 'split': changed['split'] = 'test'
            if mode == 'id': changed['id'] = old['id']
            if mode == 'parent': changed['group_id'] = old['group_id']
            if mode == 'visible':
                changed['state'], changed['question'] = copy.deepcopy(old['state']), old['question']
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, pattern):
                preparation.validate_train_blocks({'prepared_v7': [old], 'frontier_v8': [changed]}, lock)

    def test_existing_output_fails_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(preparation, 'load_lock') as load:
            with self.assertRaises(FileExistsError):
                preparation.prepare('absent', 'absent', directory, 'absent')
            load.assert_not_called()


class CommittedSourceFixtures(unittest.TestCase):
    def test_scope_checks_bound_files_only_and_rejects_dirty_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in preparation.SOURCE_FILES:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'committed\n')
            calls = []
            def git(argv, **kwargs):
                calls.append(argv)
                if argv[2] == '--show-toplevel':return str(root)+'\n'
                return 'a'*40+'\n' if argv[1] == 'rev-parse' else b'committed\n'
            with patch.object(preparation, 'ROOT', root), patch.object(preparation.subprocess, 'check_output', side_effect=git):
                self.assertEqual(preparation.committed_source()['commit'], 'a'*40)
                (root/'jev/train.py').write_bytes(b'dirty\n')
                with self.assertRaisesRegex(ValueError, 'Uncommitted'):
                    preparation.committed_source()
            self.assertTrue(all(argv[1] in ('show', 'rev-parse') for argv in calls))

    def test_inherited_git_context_cannot_redirect_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root, foreign = Path(directory)/'intended', Path(directory)/'foreign'
            for name in preparation.SOURCE_FILES:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'intended source\n')
            calls = []
            def git(argv, **kwargs):
                environment = kwargs.get('env', os.environ)
                redirected = any(name.startswith('GIT_') for name in environment)
                calls.append((argv, redirected))
                if argv[2] == '--show-toplevel':return str(foreign if redirected else root)+'\n'
                if argv[1] == 'rev-parse':return ('b' if redirected else 'a')*40+'\n'
                return b'foreign source\n' if redirected else b'intended source\n'
            inherited = {'GIT_DIR': str(foreign/'.git'), 'GIT_WORK_TREE': str(foreign),
                         'GIT_COMMON_DIR': str(foreign/'.git'), 'GIT_CONFIG_COUNT': '1',
                         'GIT_CONFIG_KEY_0': 'core.worktree', 'GIT_CONFIG_VALUE_0': str(foreign)}
            with patch.dict(os.environ, inherited), patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation.subprocess, 'check_output', side_effect=git):
                result = preparation.committed_source()
            self.assertEqual(result['commit'], 'a'*40)
            self.assertEqual(result['files_sha256'], {name: preparation.sha(b'intended source\n')
                                                     for name in preparation.SOURCE_FILES})
            self.assertFalse(any(redirected for _, redirected in calls))

    def test_wrong_toplevel_stops_before_reading_commit_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation.subprocess, 'check_output', return_value=str(root/'foreign')) as git:
                with self.assertRaisesRegex(ValueError, 'checkout differs'):
                    preparation.committed_source()
            self.assertEqual(git.call_count, 1)

    def test_git_show_uses_captured_commit_when_head_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in preparation.SOURCE_FILES:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'birth A\n')
            captured, next_head = 'a'*40, 'b'*40
            calls = []
            def git(argv, **kwargs):
                calls.append(argv)
                if argv[2] == '--show-toplevel':return str(root)+'\n'
                if argv[1] == 'rev-parse':return captured+'\n'
                # HEAD has advanced after capture; reading it would mix identities.
                return b'birth A\n' if argv[2].startswith(captured+':') else b'birth B\n'
            with patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation.subprocess, 'check_output', side_effect=git):
                result = preparation.committed_source()
            self.assertEqual(result['commit'], captured)
            self.assertNotEqual(captured, next_head)
            self.assertEqual([argv[2] for argv in calls if argv[1] == 'show'],
                             [captured+':'+name for name in preparation.SOURCE_FILES])

    def test_untracked_source_cannot_claim_head_provenance(self):
        with patch.object(preparation.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(128, 'git')):
            with self.assertRaisesRegex(ValueError, 'must be committed'):
                preparation.committed_source()


if __name__ == '__main__':
    unittest.main()
