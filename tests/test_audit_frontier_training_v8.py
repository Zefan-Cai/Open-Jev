import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_frontier_training_v8 import (SOURCE_FILES, audit_inputs,
    confirm_clones, diagnose, digest, legacy_account_core, legacy_account_query,
    safe_schema, sha, verify_committed_source, verify_stage)


def row(identity, split, state, *, group=None, question='Apply the visible exact rule.', kind='choice'):
    options = ['below', 'equal', 'above'] if kind == 'choice' else ['no', 'yes']
    return {'id': identity, 'group_id': group or identity, 'split': split,
        'source': 'hand-authored-fixture', 'state': state, 'question': question, 'kind': kind,
        'options': options, 'target': [0., 1., 0.] if kind == 'choice' else [0., 1.],
        'metadata': {'family': 'policy', 'template_id': 'fixture-'+split,
            'provenance': {'type': 'synthetic', 'license': 'CC0-1.0', 'split_policy': 'fixture-only',
                'generator_version': 'fixture', 'seed': 1, 'group_index': 1, 'variant': 0}}}


def compare(left, right):
    return {'task': 'integer_compare', 'left_cents': left, 'right_cents': right,
            'policy': 'Compare the supplied integers exactly.'}


def ledger(credit, debit):
    return {'task': 'balance', 'currency': 'USD', 'ledger': [
        {'direction': 'credit', 'amount_cents': credit}, {'direction': 'debit', 'amount_cents': debit}],
        'policy': 'Add credits and subtract debits.'}


def account():
    return {'account': 'subject', 'events': [
        {'account': 'subject', 'sequence': 1, 'status': 'active'},
        {'account': 'other', 'sequence': 3, 'status': 'closed'}],
        'rule': 'Use the highest sequence for the specified account.'}


def lines(rows):
    return ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(',', ':'))+'\n' for r in rows).encode()


class InputIsolationTests(unittest.TestCase):
    def test_inherited_typed_query_clone_blocks_and_exports_no_old_body(self):
        a = row('train', 'train', account(), question="What is this account's current status?")
        b = row('PRIVATE_OLD_ID', 'test', copy.deepcopy(a['state']),
                question="Resolve the case from the supplied facts only. What is this account's current status?")
        b['state']['account'] = 'renamed'; b['state']['events'][0]['account'] = 'renamed'
        b['state']['events'].reverse()
        result = diagnose([a], [], [b])
        self.assertEqual(result['status'], 'independent_frontier_training_inputs_blocked')
        self.assertEqual(result['typed_clone_confirmation']['confirmed_typed_account_clone_fingerprints'], 1)
        self.assertNotIn('PRIVATE_OLD_ID', json.dumps(result))
        self.assertNotIn('Use the highest', json.dumps(result))
        self.assertNotIn('target', json.dumps(result))

    def test_subject_edges_are_preserved_independently(self):
        first, second = account(), account()
        second['events'][0]['account'], second['events'][1]['account'] = 'other', 'subject'
        self.assertNotEqual(legacy_account_core(first), legacy_account_core(second))
        a = row('a', 'train', first, question="What is this account's current status?")
        b = row('b', 'test', second, question=a['question'])
        confirmation, _ = confirm_clones([a], [b])
        self.assertEqual(confirmation['confirmed_typed_account_clone_fingerprints'], 0)

    def test_noul_query_outcome_and_negation_cannot_be_ignored(self):
        a = row('a', 'train', account(), kind='noul',
            question="Does the supplied policy establish the outcome 'active'? What is this account's current status?")
        b = copy.deepcopy(a); b['question'] = a['question'].replace("'active'", "'closed'")
        self.assertNotEqual(legacy_account_query(a), legacy_account_query(b))
        b['question'] = "Does the supplied policy NOT establish the outcome 'active'? What is this account's current status?"
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            legacy_account_query(b)

    def test_unreviewed_question_does_not_claim_proven_semantic_clone(self):
        a = row('a', 'train', account(), question="What is this account's current status?")
        b = row('b', 'test', account(), question='What was the first historical status?')
        confirmation, _ = confirm_clones([a], [b])
        self.assertEqual(confirmation['confirmed_typed_account_clone_fingerprints'], 0)
        self.assertEqual(diagnose([a], [], [b])['status'], 'independent_frontier_training_inputs_blocked')

    def test_option_rotation_and_invisible_metadata_do_not_disguise_compiled_copy(self):
        a = row('a', 'train', compare(4, 9))
        b = copy.deepcopy(a); b.update(id='b', group_id='b', split='test')
        b['options'].reverse(); b['metadata']['ancestry_hash'] = 'forged'
        result = diagnose([a], [b], [])
        self.assertGreater(result['mixed_train_vs_new_reserved']['compiled_visible']['matched_fingerprints'], 0)

    def test_full_numeric_menu_origin_changes_block_but_shared_atom_alone_does_not(self):
        train = [row('a', 'train', compare(1, 2), group='A'), row('b', 'train', ledger(7, 2), group='A')]
        cloned = [row('c', 'test', compare(1001, 1002), group='B'), row('d', 'test', ledger(700, 200), group='B')]
        result = diagnose(train, cloned, [])
        self.assertEqual(result['mixed_train_vs_new_reserved']['whole_parent_scaffold']['matched_fingerprints'], 1)
        different_menu = [cloned[0], row('e', 'test', {'delivered_at': '2026-01-01T00:00:00Z',
            'request_received_at': '2026-01-01T02:00:00Z', 'return_window_seconds': 3600,
            'exception_approved': True}, group='B')]
        result = diagnose(train, different_menu, [])
        self.assertEqual(result['status'], 'independent_frontier_training_inputs_passed')
        self.assertGreater(result['mixed_train_vs_new_reserved']['magnitude_time_scaffold']['matched_fingerprints'], 0)

    def test_parent_identifier_cannot_cross_splits_with_different_facts(self):
        result = diagnose([row('a', 'train', compare(1, 2), group='same')],
                          [row('b', 'test', ledger(9, 2), group='same')], [])
        self.assertIn('mixed_train_vs_new_reserved', result['failed_checks'])

    def test_reserved_to_reserved_collision_is_checked(self):
        a = row('a', 'calibration', compare(3, 7))
        b = row('b', 'ood', compare(3, 7))
        result = diagnose([row('train', 'train', ledger(5, 1))], [a, b], [])
        self.assertIn('new_reserved_cross_split', result['failed_checks'])

    def test_parent_closure_excludes_unmatched_siblings_without_label_selection(self):
        a = row('a', 'train', account(), group='parent', question="What is this account's current status?")
        sibling = row('sibling', 'train', compare(17, 19), group='parent')
        old = row('old', 'test', account(), question=a['question'])
        first = diagnose([a, sibling], [], [old])
        old['target'] = [1., 0., 0.]; old['metadata']['hidden_gold'] = 'changed'
        second = diagnose([a, sibling], [], [old])
        self.assertEqual(first, second)
        self.assertEqual(first['quarantine_proposal']['whole_parent_rows'], 2)

    def test_preserved_full_receipt_to_exact_declared_derivative_passes(self):
        old = [row('a', 'train', account(), group='parent', question="What is this account's current status?"),
               row('sibling', 'train', compare(88, 91), group='parent')]
        new = [row('new', 'train', ledger(9, 2))]
        reserved = row('reserved', 'calibration', compare(16, 29))
        observed = [row('old', 'test', account(), question=old[0]['question'])]
        splits = {s: new if s == 'train' else [reserved] if s == 'calibration' else []
                  for s in ('train', 'calibration', 'validation', 'test', 'ood')}
        inputs = {'old_train': old, 'old_raw': lines(old), 'new_rows': splits,
            'new_raw': {s: lines(rows) for s, rows in splits.items()}, 'observed': observed,
            'input_bindings': {'fixture_only': True}, 'train_summary': safe_schema(old+new),
            'retained_counts': {}, 'watched': {}}
        full, _ = audit_inputs(inputs)
        self.assertEqual(full['status'], 'independent_frontier_training_inputs_blocked')
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/'full.json'; receipt.write_text(json.dumps(full))
            declaration = {'status': 'independent_whole_parent_exclusion_declared',
                'group_hash_algorithm': 'sha256_canonical_json', 'input_bindings': full['input_bindings'],
                'audit_source_sha256': full['audit_source_sha256'], 'blocked_full_preflight_sha256': sha(receipt.read_bytes()),
                'excluded_old_train_group_sha256': full['quarantine_proposal']['whole_old_train_group_sha256'],
                'excluded_old_train_rows': 2, 'train_summary': safe_schema(new)}
            selected, primary = audit_inputs(inputs, exclusion=declaration,
                exclusion_sha256=sha(json.dumps(declaration).encode()), blocked_receipt=receipt)
            self.assertEqual(selected['status'], 'independent_frontier_training_inputs_passed')
            self.assertEqual(primary['train'], lines(new))
            self.assertEqual(selected['train_blocks']['prepared_v7']['rows'], 0)
            declaration['excluded_old_train_group_sha256'] = []
            with self.assertRaisesRegex(ValueError, 'whole-parent closure'):
                audit_inputs(inputs, exclusion=declaration, exclusion_sha256='0'*64, blocked_receipt=receipt)


class StageIsolationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'mixture'; root.mkdir()
        train = [row('a', 'train', compare(2, 3)), row('b', 'train', ledger(7, 2))]
        primary = {s: lines(train if s == 'train' else [row(s, s, compare(11, 13))])
                   for s in ('train', 'calibration', 'validation', 'test', 'ood')}
        retained = {'observed-regression/v4/test.jsonl': lines([row('private-fixture', 'test', compare(5, 6))])}
        summary = safe_schema(train)
        blocks = {'prepared_v7': {'rows': 1, 'groups': 1, 'ordered_row_ids_sha256': digest(['a']), 'ordered_rows_sha256': digest(train[:1])},
            'frontier_v8': {'rows': 1, 'groups': 1, 'ordered_row_ids_sha256': digest(['b']), 'ordered_rows_sha256': digest(train[1:])}}
        preflight = {'status': 'independent_frontier_training_inputs_passed', 'input_bindings': {'fixture_only': True},
            'train_summary': summary, 'audit_source_sha256': 'a'*64,
            'ordered_train_row_ids_sha256': digest(['a', 'b']), 'ordered_train_rows_sha256': digest(train),
            'train_group_ids_sha256': digest(['a', 'b']), 'train_blocks': blocks,
            'retained_regression_rows': {next(iter(retained)): 1}, 'selected_observed_rows': 1,
            'full_observed_rows': 2, 'new_reserved_counts': {s: 1 for s in primary if s != 'train'}}
        receipt = Path(directory)/'preflight.json'; receipt.write_text(json.dumps(preflight))
        source = {'commit': 'b'*40, 'files_sha256': {name: 'a'*64 for name in SOURCE_FILES}}
        manifest = {'status': 'provisional_materialized_not_training_plan_frozen',
            'configuration': {'metadata_or_id_rewrites': False},
            'training_mixture_frozen': False, 'initializer_frozen': False,
            'steps_frozen': False, 'comparison_protocol_frozen': False, 'model_calls': 0, 'training_runs': 0,
            'reserved_byte_copy_only': True, 'committed_source': source,
            'materializer_source_sha256': 'a'*64, 'schema_validator_sha256': 'a'*64,
            'files_sha256': {s+'.jsonl': sha(raw) for s, raw in primary.items()},
            'retained_files_sha256': {name: sha(raw) for name, raw in retained.items()},
            'summary': copy.deepcopy(summary), 'input_bindings': preflight['input_bindings'],
            'independent_preflight_sha256': sha(receipt.read_bytes()),
            'train_membership': {'ordered_row_ids_sha256': digest(['a', 'b']), 'ordered_rows_sha256': digest(train),
                'group_ids_sha256': digest(['a', 'b']), 'blocks': copy.deepcopy(blocks)},
            'selected_observed_rows': {'v4': {'test': 1}}, 'selected_observed_total': 1,
            'full_observed_screen_total': 2, 'reserved_counts_previously_audited_not_parsed_here': preflight['new_reserved_counts']}
        for name, raw in {**{s+'.jsonl': raw for s, raw in primary.items()}, **retained}.items():
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        (root/'manifest.json').write_text(json.dumps(manifest))
        return root, preflight, primary, retained, receipt, manifest

    def verify(self, fixture):
        root, preflight, primary, retained, receipt, _ = fixture
        with patch('scripts.audit_frontier_training_v8.verify_committed_source'):
            return verify_stage(root, preflight, primary, retained, receipt)

    def test_actual_bytes_and_order_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.verify(self.fixture(directory))['status'], 'independent_frontier_training_stage_passed')

    def test_reordered_or_mutated_row_is_rejected_even_if_manifest_self_consistent(self):
        for mutate in ('order', 'metadata'):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory); root, _, primary, _, _, manifest = f
                rows = [json.loads(line) for line in primary['train'].splitlines()]
                if mutate == 'order': rows.reverse()
                else: rows[0]['metadata']['rewritten'] = True
                raw = lines(rows); (root/'train.jsonl').write_bytes(raw)
                manifest['files_sha256']['train.jsonl'] = sha(raw)
                (root/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'primary hash'):
                    self.verify(f)

    def test_missing_retained_or_changed_bytes_rejected(self):
        for remove in (True, False):
            with self.subTest(remove=remove), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory); path = f[0]/next(iter(f[3]))
                if remove: path.unlink()
                else: path.write_bytes(b'altered\n')
                with self.assertRaises(ValueError): self.verify(f)

    def test_unexpected_nonjsonl_jsonl_symlink_and_fifo_rejected_before_read(self):
        for kind in ('text', 'jsonl', 'symlink', 'fifo'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory); extra = f[0]/('extra.jsonl' if kind == 'jsonl' else 'extra')
                if kind == 'symlink': extra.symlink_to(f[0]/'train.jsonl')
                elif kind == 'fifo': os.mkfifo(extra)
                else: extra.write_text('unexpected')
                with self.assertRaisesRegex(ValueError, 'Unexpected stage'):
                    self.verify(f)

    def test_declared_blocks_counts_and_membership_cannot_be_forged(self):
        for field in ('blocks', 'counts', 'membership', 'summary'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory); manifest = f[-1]
                if field == 'blocks': manifest['train_membership']['blocks']['prepared_v7']['rows'] += 1
                elif field == 'counts': manifest['selected_observed_total'] += 1
                elif field == 'membership': manifest['train_membership']['ordered_row_ids_sha256'] = '0'*64
                else: manifest['summary']['records'] += 1
                (f[0]/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): self.verify(f)

    def test_promotion_model_calls_or_executable_argv_is_rejected(self):
        for field in ('initializer_frozen', 'model_calls', 'argv'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory); manifest = f[-1]
                manifest[field] = ['python', '-m', 'jev.train'] if field == 'argv' else True if field.endswith('frozen') else 1
                (f[0]/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): self.verify(f)

    def test_blocked_preflight_fails_before_directory_read(self):
        with self.assertRaisesRegex(ValueError, 'Blocked'):
            verify_stage('/does/not/exist', {'status': 'independent_frontier_training_inputs_blocked'}, {}, {}, '/absent')

    def test_changed_input_receipt_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory); f[4].write_text('{}')
            with self.assertRaisesRegex(ValueError, 'preflight binding'): self.verify(f)


class CommittedSourceTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        content = b'hand-authored committed source\n'
        for name in SOURCE_FILES:
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        return root, content, {'commit': 'b'*40, 'files_sha256': {name: sha(content) for name in SOURCE_FILES}}

    def test_every_git_variable_scrubbed_and_exact_commit_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root, content, source = self.fixture(directory)
            def git(args, **kwargs):
                self.assertFalse(any(k.startswith('GIT_') for k in kwargs['env']))
                if args[-1] == '--show-toplevel': return (str(root)+'\n').encode()
                self.assertTrue(args[-1].startswith(source['commit']+':'))
                return content
            with patch('scripts.audit_frontier_training_v8.ROOT', root), patch(
                'scripts.audit_frontier_training_v8.subprocess.check_output', side_effect=git), patch.dict(os.environ,
                {'GIT_DIR': '/foreign', 'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.worktree',
                 'GIT_CONFIG_VALUE_0': '/foreign', 'GIT_OBJECT_DIRECTORY': '/foreign'}):
                verify_committed_source(source, sha(content))

    def test_wrong_repository_root_or_source_bytes_rejected(self):
        for wrong_root in (True, False):
            with self.subTest(wrong_root=wrong_root), tempfile.TemporaryDirectory() as directory:
                root, content, source = self.fixture(directory)
                if not wrong_root: (root/SOURCE_FILES[0]).write_bytes(b'changed')
                def git(args, **kwargs):
                    if args[-1] == '--show-toplevel': return b'/foreign\n' if wrong_root else (str(root)+'\n').encode()
                    return content
                with patch('scripts.audit_frontier_training_v8.ROOT', root), patch(
                    'scripts.audit_frontier_training_v8.subprocess.check_output', side_effect=git):
                    with self.assertRaisesRegex(ValueError, 'root|implementation'):
                        verify_committed_source(source, sha(content))


if __name__ == '__main__':
    unittest.main()
