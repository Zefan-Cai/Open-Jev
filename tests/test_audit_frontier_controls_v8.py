"""Independent hand-authored oracle fixtures; no generator imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_frontier_controls_v8 as oracle


class PrimitiveOracleTests(unittest.TestCase):
    def test_exact_integer_and_boolean_types(self):
        self.assertEqual(oracle.integer(-1, 'signed amount'), -1)
        self.assertEqual(oracle.integer(0, 'amount', nonnegative=True), 0)
        self.assertIs(oracle.boolean(False, 'signature'), False)
        for value in (True, 1.0, '1', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                oracle.integer(value, 'amount')
        for value in (-1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                oracle.integer(value, 'capacity', nonnegative=True)
        for value in (0, 1, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                oracle.boolean(value, 'signature')

    def test_time_uses_absolute_instants_across_offsets_and_leap_day(self):
        self.assertEqual(oracle.instant_seconds('2024-02-29T12:00:00Z'),
                         oracle.instant_seconds('2024-02-29T07:00:00-05:00'))
        self.assertEqual(oracle.instant_seconds('2024-03-01T00:00:00Z') -
                         oracle.instant_seconds('2024-02-29T23:59:59Z'), 1)
        for value in ('2023-02-29T00:00:00Z', '2024-01-01T24:00:00Z',
                      '2024-01-01T00:00:00+24:00', '2024-01-01T00:00:00',
                      '2024-01-01T00:00:00+00:60', '2024-01-01T00:60:00Z',
                      '2024-01-01T00:00:00.1Z', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                oracle.instant_seconds(value)

    def test_money_format_is_exact_for_negative_and_large_cents(self):
        self.assertEqual(oracle.cents_label(-1, 'USD'), 'USD -0.01')
        self.assertEqual(oracle.cents_label(0, 'USD'), 'USD 0.00')
        self.assertEqual(oracle.cents_label(100000001, 'EUR'), 'EUR 1000000.01')
        with self.assertRaises(ValueError):
            oracle.cents_label(True, 'USD')


def joint():
    scope = {'resource':'vault','operation':'release','currency':'USD'}
    return {'case_ref':'fixture','request':{**scope,'amount_cents':100},
            'trusted_policy':{'rules':oracle.POLICIES['joint_capacity'],'required_roles':['A','B']},
            'signed_events':[{'issuer_role':role,'verified_signature':True,'credential_scope':copy.deepcopy(scope),
                              'sequence':1,'status':'grant','capacity_cents':capacity}
                             for role,capacity in (('A',100),('B',200))]}


def authority():
    return {'case_ref':'fixture','trust_contract':oracle.POLICIES['latest_authority'],
            'request':{'department':'care','amount_cents':100,'confirmed_fraud':False},
            'required_topics':['privacy','funds'],'policy_authorities':{'privacy':'board A','funds':'board B'},
            'signed_policy_registry':[{'topic':topic,'department':'care','credential_scope':{'department':'care','topic':topic},
                                       'issuer_role':issuer,'verified_signature':True,'revision':1,'status':'active',
                                       'automatic_limit_cents':limit}
                                      for topic,issuer,limit in (('privacy','board A',100),('funds','board B',200))]}


def account():
    return {'case_ref':'fixture','rule':oracle.POLICIES['account_status'],
            'required_accounts':['A','B','C'],
            'events':[{'account':name,'sequence':1,'status':'active'} for name in ('A','B','C')]}


def department():
    return {'case_ref':'fixture','policy':oracle.POLICIES['department_route'],'policy_authority':'board',
            'request_lines':[{'department':'care','amount_cents':100,'confirmed_fraud':False},
                             {'department':'support','amount_cents':100,'confirmed_fraud':False}],
            'signed_policies':[{'department':name,'credential_scope':name,'issuer_role':'board','verified_signature':True,
                                'revision':1,'status':'active','approval_limit_cents':100}
                               for name in ('care','support')]}


def refund():
    return {'case_ref':'fixture','purchase_document_verified':True,
            'trusted_policy':{'ordered_rule':oracle.POLICIES['refund_priority'],'currency':'USD','automatic_limit_cents':200},
            'items':[{'item_ref':name,'amount_cents':100,'defect_confirmed':False,'seal_intact':True,'verified_recall':False}
                     for name in ('A','B')]}


def temporal():
    return {'case_ref':'fixture','policy':oracle.POLICIES['temporal_window'],
            'recorded_delivery_at':'2024-01-01T00:00:00Z','correction_authority':'board',
            'delivery_corrections':[{'sequence':1,'issuer_role':'board','verified_signature':True,'case_ref':'fixture',
                                     'direction':'delay','seconds':60}],
            'request_received_at':'2023-12-31T19:01:00-05:00','return_window_seconds':60,'exception_approved':False}


def numeric():
    def entries(credit,debit):
        return [{'entry_ref':'credit','direction':'credit','amount_cents':credit},
                {'entry_ref':'debit','direction':'debit','amount_cents':debit}]
    return {'case_ref':'fixture','policy':oracle.POLICIES['exact_numeric'],'currency':'USD','task':'integer_compare',
            'left_ledger':entries(101,100),'right_ledger':entries(102,100)}


class StateOracleTests(unittest.TestCase):
    def test_temporal_corrected_anchor_inclusive_endpoints_exception_and_invalid_records(self):
        state = temporal()
        self.assertEqual(oracle.outcome(state),'accept')
        state['request_received_at'] = '2024-01-01T00:02:00Z'
        self.assertEqual(oracle.outcome(state),'accept')
        state['request_received_at'] = '2024-01-01T00:02:01Z'
        self.assertEqual(oracle.outcome(state),'reject')
        state['exception_approved'] = True
        self.assertEqual(oracle.outcome(state),'review')
        state['exception_approved'] = False
        bad = copy.deepcopy(state['delivery_corrections'][0])
        bad.update(sequence=9,issuer_role='attacker',seconds=100000)
        state['delivery_corrections'].append(bad)
        self.assertEqual(oracle.outcome(state),'reject')
        self.assertEqual(oracle.facts(state)['width'],1)

    def test_temporal_advance_and_duplicate_applicable_sequences(self):
        state = temporal()
        advance = copy.deepcopy(state['delivery_corrections'][0])
        advance.update(sequence=2,direction='advance',seconds=30)
        state['delivery_corrections'].append(advance)
        state['request_received_at'] = '2024-01-01T00:00:29Z'
        self.assertEqual(oracle.outcome(state),'reject')
        state['request_received_at'] = '2024-01-01T00:00:30Z'
        self.assertEqual(oracle.outcome(state),'accept')
        state['delivery_corrections'].append(copy.deepcopy(advance))
        with self.assertRaisesRegex(ValueError,'correction sequence'):
            oracle.outcome(state)

    def test_joint_exact_capacity_scope_and_revoke_before_missing_or_amount(self):
        state = joint()
        self.assertEqual(oracle.outcome(state),'execute')
        state['request']['amount_cents'] = 101
        self.assertEqual(oracle.outcome(state),'request higher capacity')
        state['signed_events'][1]['credential_scope']['currency'] = 'EUR'
        self.assertEqual(oracle.outcome(state),'request missing consent')
        revoke = copy.deepcopy(state['signed_events'][0])
        revoke.update(sequence=2,status='revoke')
        state['signed_events'].append(revoke)
        self.assertEqual(oracle.outcome(state),'reject revoked consent')
        invalid = copy.deepcopy(revoke)
        invalid.update(sequence=3,status='grant',verified_signature=False,capacity_cents=999999)
        state['signed_events'].append(invalid)
        self.assertEqual(oracle.outcome(state),'reject revoked consent')
        state['signed_events'].reverse()
        self.assertEqual(oracle.outcome(state),'reject revoked consent')

    def test_joint_regrant_and_tied_authority(self):
        state = joint()
        revoke = copy.deepcopy(state['signed_events'][0]); revoke.update(sequence=2,status='revoke')
        regrant = copy.deepcopy(revoke); regrant.update(sequence=3,status='grant')
        state['signed_events'] += [regrant,revoke]
        self.assertEqual(oracle.outcome(state),'execute')
        state['signed_events'].append(copy.deepcopy(regrant))
        with self.assertRaisesRegex(ValueError,'role/sequence'):
            oracle.outcome(state)

    def test_topics_resolve_all_authority_before_fraud_and_ignore_invalid_high_revision(self):
        state = authority()
        self.assertEqual(oracle.outcome(state),'automatic processing')
        state['request']['amount_cents'] = 101
        self.assertEqual(oracle.outcome(state),'capacity review')
        state['request']['confirmed_fraud'] = True
        self.assertEqual(oracle.outcome(state),'fraud review')
        withdrawn = copy.deepcopy(state['signed_policy_registry'][0]); withdrawn.update(revision=2,status='withdrawn')
        state['signed_policy_registry'].append(withdrawn)
        self.assertEqual(oracle.outcome(state),'verify policy')
        invalid = copy.deepcopy(withdrawn); invalid.update(revision=9,status='active',issuer_role='attacker',automatic_limit_cents=999999)
        state['signed_policy_registry'].append(invalid)
        self.assertEqual(oracle.outcome(state),'verify policy')
        state['signed_policy_registry'].reverse()
        self.assertEqual(oracle.outcome(state),'verify policy')

    def test_topic_scope_exactness_and_duplicate_valid_revision(self):
        for field in ('department','topic'):
            state = authority()
            state['request']['confirmed_fraud'] = True
            state['signed_policy_registry'][0]['credential_scope'][field] = 'elsewhere'
            with self.subTest(field=field):
                self.assertEqual(oracle.outcome(state),'verify policy')
        state = authority()
        state['signed_policy_registry'].append(copy.deepcopy(state['signed_policy_registry'][0]))
        with self.assertRaisesRegex(ValueError,'topic/revision'):
            oracle.outcome(state)

    def test_policy_identity_normalization_ignores_mapping_key_order(self):
        state = authority()
        before = oracle.semantic_state(state)
        state['policy_authorities'] = dict(reversed(list(state['policy_authorities'].items())))
        state['signed_policy_registry'].reverse()
        self.assertEqual(oracle.outcome(state),'automatic processing')
        self.assertEqual(oracle.semantic_state(state),before)

    def test_account_closed_paused_unknown_active_priority_and_latest_status(self):
        state = account()
        self.assertEqual(oracle.outcome(state),'active')
        state['events'].pop()
        self.assertEqual(oracle.outcome(state),'unknown')
        state['events'][0]['status'] = 'paused'
        self.assertEqual(oracle.outcome(state),'paused')
        state['events'][1]['status'] = 'closed'
        self.assertEqual(oracle.outcome(state),'closed')
        state['events'].append({'account':'B','sequence':2,'status':'active'})
        self.assertEqual(oracle.outcome(state),'paused')
        state['events'].append({'account':'other','sequence':999,'status':'closed'})
        self.assertEqual(oracle.outcome(state),'paused')
        state['events'].append({'account':'B','sequence':2,'status':'closed'})
        with self.assertRaisesRegex(ValueError,'account/sequence'):
            oracle.outcome(state)

    def test_department_fraud_before_missing_or_withdrawn_policy(self):
        state = department()
        self.assertEqual(oracle.outcome(state),'approve')
        state['request_lines'][1]['amount_cents'] = 101
        self.assertEqual(oracle.outcome(state),'manager review')
        state['signed_policies'].clear()
        self.assertEqual(oracle.outcome(state),'manager review')
        state['request_lines'][0]['confirmed_fraud'] = True
        self.assertEqual(oracle.outcome(state),'security review')
        state = department()
        withdrawn = copy.deepcopy(state['signed_policies'][0]); withdrawn.update(revision=2,status='withdrawn')
        state['signed_policies'].append(withdrawn)
        self.assertEqual(oracle.outcome(state),'manager review')
        state['signed_policies'].append(copy.deepcopy(withdrawn))
        with self.assertRaisesRegex(ValueError,'department/revision'):
            oracle.outcome(state)

    def test_department_normalization_preserves_unequal_components_under_reversal(self):
        state = department()
        state['request_lines'][0]['amount_cents'] = 99
        state['request_lines'][1]['amount_cents'] = 199
        state['signed_policies'][1]['approval_limit_cents'] = 200
        before = oracle.semantic_state(state)
        state['request_lines'].reverse()
        state['signed_policies'].reverse()
        self.assertEqual(oracle.semantic_state(state),before)
        self.assertEqual(oracle.outcome(state),'approve')
        state['request_lines'][0]['amount_cents'] -= 1
        self.assertEqual(oracle.outcome(state),'approve')
        self.assertNotEqual(oracle.semantic_state(state),before)

    def test_foreign_reference_equality_is_preserved_even_when_gold_ignores_it(self):
        state = account()
        state['events'] += [{'account':'outside-one','sequence':1,'status':'active'},
                            {'account':'outside-one','sequence':2,'status':'closed'}]
        before = oracle.semantic_state(state)
        self.assertEqual(oracle.outcome(state),'active')
        state['events'][-1]['account'] = 'outside-two'
        self.assertEqual(oracle.outcome(state),'active')
        self.assertNotEqual(oracle.semantic_state(state),before)

    def test_full_typed_presentation_matcher_preserves_all_ids_and_values(self):
        state = account()
        state['events'] += [{'account':'outside-one','sequence':1,'status':'active'},
                            {'account':'outside-one','sequence':2,'status':'closed'}]
        reverse = oracle.reversed_record_state(state)
        reordered_keys = dict(reversed(list(reverse.items())))
        self.assertTrue(oracle.same_typed_state(reverse,reordered_keys))
        self.assertFalse(oracle.same_typed_state(state,reverse))
        altered = copy.deepcopy(reverse)
        altered['events'][0]['account'] = 'outside-two'
        self.assertFalse(oracle.same_typed_state(reverse,altered))
        self.assertEqual(oracle.outcome(altered),oracle.outcome(reverse))
        altered = copy.deepcopy(reverse)
        altered['events'][0]['sequence'] = True
        self.assertFalse(oracle.same_typed_state(reverse,altered))
        state = numeric()
        altered = copy.deepcopy(state)
        altered['right_ledger'][0]['entry_ref'] = altered['left_ledger'][0]['entry_ref']+'new'
        self.assertFalse(oracle.same_typed_state(state,altered))

    def test_full_matcher_accepts_consistent_renaming_and_rejects_identity_merge(self):
        state = authority()
        changed = copy.deepcopy(state)
        substitutions = {'fixture':'renamed case','privacy':'topic X','funds':'topic Y',
                         'board A':'authority X','board B':'authority Y','care':'department X'}
        def rename(value):
            if isinstance(value,dict):return {substitutions.get(key,key):rename(item) for key,item in value.items()}
            if isinstance(value,list):return [rename(item) for item in value]
            if isinstance(value,str):return substitutions.get(value,value)
            return value
        changed = rename(changed)
        changed['policy_authorities'] = dict(reversed(list(changed['policy_authorities'].items())))
        self.assertTrue(oracle.same_typed_state(state,changed))
        changed['policy_authorities']['topic Y'] = 'authority X'
        self.assertFalse(oracle.same_typed_state(state,changed))
        state = account()
        changed = copy.deepcopy(state)
        changed['required_accounts'][0] = changed['required_accounts'][1]
        self.assertFalse(oracle.same_typed_state(state,changed))

    def test_refund_recall_before_document_ineligibility_and_total_capacity(self):
        state = refund()
        self.assertEqual(oracle.outcome(state),'automatic reimbursement')
        state['items'][0]['amount_cents'] += 1
        self.assertEqual(oracle.outcome(state),'review reimbursement')
        state['purchase_document_verified'] = False
        state['items'][1]['seal_intact'] = False
        self.assertEqual(oracle.outcome(state),'reject request')
        state['items'][0]['verified_recall'] = True
        self.assertEqual(oracle.outcome(state),'recall remediation')
        state['items'][0]['verified_recall'] = 1
        with self.assertRaisesRegex(ValueError,'boolean'):
            oracle.outcome(state)

    def test_refund_canonical_first_last_annotations_survive_verified_reversal(self):
        state = refund()
        state['items'][0]['seal_intact'] = False
        row = {'id':'fixture','state':state,
               'metadata':{'condition':'reject_first_item','layout':'forward'}}
        oracle.check_condition(row,oracle.facts(state))
        original = oracle.semantic_state(state)
        original_order = oracle.semantic_state(state,sort_records=False)
        row['state']['items'].reverse()
        row['metadata']['layout'] = 'reverse'
        oracle.check_condition(row,oracle.facts(row['state']))
        self.assertEqual(oracle.outcome(row['state']),'reject request')
        self.assertEqual(original,oracle.semantic_state(row['state']))
        self.assertNotEqual(original_order,oracle.semantic_state(row['state'],sort_records=False))
        row['metadata']['layout'] = 'forward'
        with self.assertRaisesRegex(ValueError,'Condition annotation'):
            oracle.check_condition(row,oracle.facts(row['state']))

    def test_composed_signed_ledgers_and_no_boolean_or_neutral_operands(self):
        state = numeric()
        self.assertEqual(oracle.outcome(state),'below')
        state['left_ledger'][0]['amount_cents'] = 102
        self.assertEqual(oracle.outcome(state),'equal')
        state['left_ledger'][0]['amount_cents'] = 103
        self.assertEqual(oracle.outcome(state),'above')
        state['left_ledger'][0]['amount_cents'] = True
        with self.assertRaisesRegex(ValueError,'integer'):
            oracle.outcome(state)
        state = numeric()
        state['left_ledger'][0]['amount_cents'] = 0
        with self.assertRaisesRegex(ValueError,'positive'):
            oracle.outcome(state)
        state = numeric()
        state['left_ledger'] += [{'entry_ref':'paired credit','direction':'credit','amount_cents':7},
                                 {'entry_ref':'paired debit','direction':'debit','amount_cents':7}]
        with self.assertRaisesRegex(ValueError,'cancelling operand pairs'):
            oracle.outcome(state)

    def test_visible_policy_and_proposition_gold_ignore_metadata_conditions(self):
        state = joint()
        row = {'state':state,'kind':'choice','question':oracle.CHOICE_QUESTIONS[0],
               'options':list(reversed(oracle.OPTIONS['joint_capacity'])),
               'metadata':{'scenario_family':'refund_priority','condition':'fabricated'}}
        self.assertEqual(oracle.gold(row),'execute')
        row.update(kind='noul',options=['no','yes'],question=oracle.NOUL_TEMPLATE.format(proposed_outcome='execute'))
        self.assertEqual(oracle.gold(row),'yes')
        row['question'] = oracle.NOUL_TEMPLATE.format(proposed_outcome='request missing consent')
        self.assertEqual(oracle.gold(row),'no')
        row['state']['trusted_policy']['rules'] = 'Unreviewed policy; always execute'
        with self.assertRaisesRegex(ValueError,'reviewed visible policy'):
            oracle.gold(row)

    def test_old_directory_and_symlink_are_rejected_before_any_row_read(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('read forbidden')):
            with self.assertRaisesRegex(ValueError,'new original'):
                oracle.audit_directory(Path('/not-read/boundary-controls-v7-old'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frontier-controls-v8-alias'
            path.symlink_to(Path(directory)/'old-dataset',target_is_directory=True)
            with patch.object(Path,'read_bytes',side_effect=AssertionError('read forbidden')):
                with self.assertRaisesRegex(ValueError,'new original'):
                    oracle.audit_directory(path)

    def test_provisional_split_hash_rejected_before_row_json_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)/'frontier-controls-v8-fixture'
            dataset.mkdir()
            filenames = [split+'.jsonl' for split in oracle.SPLIT_WIDTHS]
            for name in filenames:
                (dataset/name).write_bytes(b'not a JSON row\n')
            (dataset/'manifest.json').write_text(json.dumps({
                'configuration':{'generator_version':oracle.VERSION},
                'files_sha256':{name:'0'*64 for name in filenames}}))
            with self.assertRaisesRegex(ValueError,'checksum differs before parsing'):
                oracle.audit_directory(dataset)

    def test_canonical_balance_candidates_and_visible_proposition(self):
        state = numeric()
        state['task'] = 'balance'
        state['ledger'] = state.pop('left_ledger')
        state.pop('right_ledger')
        state['ledger'][0]['amount_cents'] = 99
        row = {'state':state,'kind':'choice','question':oracle.CHOICE_QUESTIONS[0],
               'options':['USD -0.01','USD 0.00','USD 0.01','USD -1.00','USD 1.00']}
        self.assertEqual(oracle.gold(row),'USD -0.01')
        row['options'][1] = 'USD -0.00'
        with self.assertRaisesRegex(ValueError,'canonical'):
            oracle.gold(row)
        row.update(kind='noul',options=['no','yes'],
                   question=oracle.NOUL_TEMPLATE.format(proposed_outcome='USD -0.01'))
        self.assertEqual(oracle.gold(row),'yes')
        row['question'] = oracle.NOUL_TEMPLATE.format(proposed_outcome='USD 0.01')
        self.assertEqual(oracle.gold(row),'no')


if __name__ == '__main__':
    unittest.main()
