"""Regressions for heldout protection beyond exact IDs and option ordering."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_boundary_training_v7 as preparation


def row(split='test', identity='reserved'):
    return {'id': identity+'/row', 'group_id': identity+'/group', 'split': split,
        'source': 'original', 'state': {'case_ref': identity,
            'request': {'resource': '/vault/'+identity, 'amount_cents': 501},
            'signed_events': [{'resource': '/vault/'+identity, 'capacity_cents': 500},
                              {'resource': '/vault/'+identity+'-other', 'capacity_cents': 999}]},
        'question': 'Apply the exact authority and capacity rule.', 'kind': 'choice',
        'options': ['execute', 'review'], 'target': [0., 1.],
        'metadata': {'entity_ids': [identity]}}


class BoundaryPreparationTest(unittest.TestCase):
    def test_original_train_and_all_heldouts_are_reserved_against_new_copies(self):
        for split in preparation.SPLITS:
            old = row(split)
            new = copy.deepcopy(old)
            new.update(id='new/row', group_id='new/group', split='train', source='new')
            with self.assertRaisesRegex(ValueError, 'overlaps an original'):
                preparation.reserved_overlap([new], [old])

    def test_identifier_renaming_and_options_cannot_launder_old_test(self):
        old = row()
        new = row('train', 'fresh-name')
        new['options'].reverse()
        new['target'].reverse()
        self.assertEqual(preparation.identity_normalized_context(old),
                         preparation.identity_normalized_context(new))
        with self.assertRaisesRegex(ValueError, 'overlaps an original'):
            preparation.reserved_overlap([new], [old])

    def test_normalization_keeps_scope_relations_and_one_cent_facts(self):
        old = row()
        new = row('train', 'fresh-name')
        new['state']['request']['amount_cents'] += 1
        result = preparation.reserved_overlap([new], [old])
        self.assertEqual(result['matches'], 0)
        changed_scope = copy.deepcopy(old)
        changed_scope['state']['signed_events'][0]['resource'] += '-other'
        self.assertNotEqual(preparation.identity_normalized_context(old),
                            preparation.identity_normalized_context(changed_scope))

    def test_aliases_and_unordered_lists_cannot_launder_a_reserved_state(self):
        old = row()
        new = row('train', 'fresh-name')
        new['state']['signed_events'][1]['resource'] = '/completely-unrelated'
        new['state']['signed_events'].reverse()
        new['question'] = 'Please select the route!'
        new['kind'] = 'noul'
        with self.assertRaisesRegex(ValueError, 'overlaps an original'):
            preparation.reserved_overlap([new], [old])
        old['state']['ledger'] = [{'direction': 'credit', 'amount_cents': 501},
                                 {'direction': 'debit', 'amount_cents': 1}]
        new = row('train', 'fresh-name')
        new['state']['ledger'] = list(reversed(old['state']['ledger']))
        with self.assertRaisesRegex(ValueError, 'overlaps an original'):
            preparation.reserved_overlap([new], [old])

    def test_same_split_duplicate_or_identifier_alias_is_rejected(self):
        a, b = row('train', 'first'), row('train', 'second')
        with self.assertRaisesRegex(ValueError, 'repeats within the new candidate'):
            preparation.reserved_overlap([a, b], [])

    def test_existing_output_and_uncommitted_source_fail_before_loading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(preparation, 'committed_source', side_effect=ValueError('uncommitted')) as source:
                with self.assertRaises(FileExistsError):
                    preparation.prepare('v4', 'v5', 'v6', 'v7', tmp, 'checkpoint', 'training')
                source.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'uncommitted'):
                    preparation.prepare('v4', 'v5', 'v6', 'v7', Path(tmp)/'fresh', 'checkpoint', 'training')
                self.assertFalse((Path(tmp)/'fresh').exists())

    def test_candidate_split_changed_after_independent_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in preparation.SPLITS:
                value = row(split)
                value['source'] = preparation.VERSION
                (root/(split+'.jsonl')).write_text(json.dumps(value)+'\n')
            hashes = {s+'.jsonl': preparation._file_sha256(root/(s+'.jsonl')) for s in preparation.SPLITS}
            generator_hash = preparation._file_sha256(preparation.ROOT/'jev/boundary_controls_v7.py')
            manifest = {'files_sha256': hashes, 'generator_sha256': generator_hash,
                        'configuration': {'generator_version': preparation.VERSION}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            digest = preparation._file_sha256(root/'manifest.json')
            visible = {'status': 'no_detected_overlap', 'matched_rows': 0,
                       'dataset_manifest_sha256': digest, 'split_sha256': hashes,
                       'screen_script_sha256': preparation._file_sha256(preparation.ROOT/'scripts/screen_boundary_inputs_v7.py'),
                       'compile_api_sha256': preparation._file_sha256(preparation.ROOT/'jev/api.py'),
                       'lexical_screen_sha256': preparation._file_sha256(preparation.ROOT/'scripts/screen_training_overlap.py')}
            (root/'visible.json').write_text(json.dumps(visible))
            lock = {'dataset_manifest_sha256': digest, 'split_sha256': hashes,
                    'generator_sha256': generator_hash, 'visible_overlap_audit_sha256': preparation._file_sha256(root/'visible.json')}
            (root/'lock.json').write_text(json.dumps(lock))
            audit = {'status': 'validated_no_model_run', 'schema': {'splits': preparation.COUNTS},
                     'manifest_sha256': digest, 'files_sha256': hashes,
                     'validator_sha256': preparation._file_sha256(preparation.ROOT/'scripts/audit_boundary_controls_v7.py')}
            (root/'audit.json').write_text(json.dumps(audit))
            lock['independent_audit_sha256'] = preparation._file_sha256(root/'audit.json')
            (root/'lock.json').write_text(json.dumps(lock))

            def audit_then_mutate(directory):
                (Path(directory)/'train.jsonl').write_text('{}\n')
                return audit

            with patch.object(preparation, 'CANDIDATE_LOCK', root/'lock.json'), \
                    patch.object(preparation, 'VISIBLE_AUDIT', root/'visible.json'), \
                    patch.object(preparation, 'INDEPENDENT_AUDIT', root/'audit.json'), \
                    patch('scripts.audit_boundary_controls_v7.audit_directory', side_effect=audit_then_mutate):
                with self.assertRaisesRegex(ValueError, 'changed after independent audit'):
                    preparation.read_candidate(root)


if __name__ == '__main__':
    unittest.main()
