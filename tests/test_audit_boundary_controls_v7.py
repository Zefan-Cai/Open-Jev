"""Independent hand-authored gold fixtures; no generator/oracle imports."""
import copy
from datetime import datetime, timedelta, timezone
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev.data import SPLITS, _write_dataset
from scripts import audit_boundary_controls_v7 as audit


def joint_state():
    request = dict(resource='vault',operation='release',currency='USD',amount_cents=100)
    scope = {key:request[key] for key in ('resource','operation','currency')}
    return dict(request=request,trusted_policy=dict(required_roles=['A','B'],rules=audit.CONTRACTS['joint_capacity']),
        signed_events=[dict(issuer_role=role,verified_signature=True,credential_scope=copy.deepcopy(scope),
            sequence=1,status='grant',capacity_cents=100+i*100) for i,role in enumerate(('A','B'))])


def authority_state():
    policy = dict(department='care',credential_scope='care',issuer_role='board',verified_signature=True,
                  revision=2,status='active',automatic_limit_cents=100)
    return dict(request=dict(department='care',amount_cents=100,confirmed_fraud=False),policy_authority='board',
        signed_policy_registry=[{**policy,'revision':1,'automatic_limit_cents':200},policy],
        trust_contract=audit.CONTRACTS['latest_authority'])


def row(state,family,answer):
    options = list(audit.CHOICE_OPTIONS.get(family,{'below','equal','above'}))
    return dict(id='fixture/row',group_id='fixture/group',split='test',source=audit.VERSION,state=state,
        question='Apply the supplied exact rule to the recorded facts. Which stated outcome follows?',kind='choice',
        options=options,target=[float(label == answer) for label in options],metadata=dict(scenario_family=family,proposed_outcome=None))


class IndependentGoldTests(unittest.TestCase):
    def test_gregorian_offset_manual_epoch_fixtures(self):
        epoch = audit.instant_seconds('1970-01-01T00:00:00Z')
        fixtures = [('2031-06-15T05:29:59+05:30',1939247999),('2031-06-15T05:30:00+05:30',1939248000),
            ('2000-02-28T23:30:00-00:30',951782400),('2000-03-01T09:30:00+09:30',951868800),
            ('2100-03-01T01:00:00+01:00',4107542400),('2400-02-29T09:00:00+09:00',13574563200)]
        for timestamp,seconds in fixtures:
            with self.subTest(timestamp=timestamp):self.assertEqual(audit.instant_seconds(timestamp)-epoch,seconds)
        for invalid in ('2100-02-29T00:00:00Z','2030-01-01T00:00:00','2030-01-01T00:00:00+24:00'):
            with self.assertRaises(ValueError):audit.instant_seconds(invalid)

    def test_temporal_closed_endpoints_and_exception_precedence(self):
        state=dict(delivered_at='2100-02-28T23:00:00Z',request_received_at='2100-03-01T01:00:00+01:00',
            return_window_seconds=3600,exception_approved=False)
        self.assertEqual(audit.temporal_outcome(state),'accept')
        state['request_received_at']='2100-03-01T01:00:01+01:00'
        self.assertEqual(audit.temporal_outcome(state),'reject')
        state['exception_approved']=True
        self.assertEqual(audit.temporal_outcome(state),'review')
        state['exception_approved']='false'
        with self.assertRaises(ValueError):audit.temporal_outcome(state)

    def test_integer_ledger_cancellation_negative_and_one_cent_without_float(self):
        entries=[dict(direction='credit',amount_cents=100),dict(direction='debit',amount_cents=101),
                 dict(direction='credit',amount_cents=2**60),dict(direction='debit',amount_cents=2**60)]
        self.assertEqual(audit.ledger_total(entries),-1)
        self.assertEqual(audit.cents_label(-1,'USD'),'USD -0.01')
        self.assertEqual(audit.cents_label(2**60+1,'USD'),'USD 11529215046068469.77')
        self.assertEqual(audit.ledger_total(list(reversed(entries))),-1)
        for value in (True,1.,-1):
            with self.assertRaises(ValueError):audit.ledger_total([dict(direction='credit',amount_cents=value)])
        for left,right,expected in ((-101,-100,'below'),(-100,-100,'equal'),(-100,-101,'above'),(101,100,'above')):
            self.assertEqual(audit.numeric_outcome(dict(task='integer_compare',left_cents=left,right_cents=right)),expected)

    def test_joint_latest_valid_scope_and_minimum_capacity(self):
        state=joint_state();self.assertEqual(audit.joint_outcome(state),'execute')
        state['request']['amount_cents']=101
        self.assertEqual(audit.joint_outcome(state),'request higher capacity')
        state['request']['amount_cents']=100
        for field in ('resource','operation','currency'):
            changed=copy.deepcopy(state);changed['signed_events'][0]['credential_scope'][field]+='-wrong'
            self.assertEqual(audit.joint_outcome(changed),'request missing consent')
        signed=copy.deepcopy(state['signed_events'][0])
        state['signed_events'] += [{**signed,'sequence':2,'status':'revoke'},
                                   {**signed,'sequence':99,'verified_signature':False,'capacity_cents':999}]
        self.assertEqual(audit.joint_outcome(state),'reject revoked consent')
        state['signed_events'].append({**signed,'sequence':3})
        state['signed_events'].reverse()
        self.assertEqual(audit.joint_outcome(state),'execute')
        state['signed_events'].append({**signed,'sequence':3,'status':'revoke'})
        with self.assertRaises(ValueError):audit.joint_outcome(state)

    def test_revocation_precedes_missing_consent(self):
        state=joint_state();state['signed_events']=state['signed_events'][:1]
        state['signed_events'][0]['status']='revoke'
        self.assertEqual(audit.joint_outcome(state),'reject revoked consent')

    def test_policy_withdrawal_and_absence_precede_fraud_no_fallback(self):
        state=authority_state();self.assertEqual(audit.authority_outcome(state),'automatic processing')
        state['request']['amount_cents']=101
        self.assertEqual(audit.authority_outcome(state),'capacity review')
        state['request']['confirmed_fraud']=True
        self.assertEqual(audit.authority_outcome(state),'fraud review')
        current=state['signed_policy_registry'][1]
        state['signed_policy_registry'] += [{**current,'revision':3,'status':'withdrawn'},
            {**current,'revision':999,'verified_signature':False,'automatic_limit_cents':1000}]
        self.assertEqual(audit.authority_outcome(state),'verify policy')
        state['signed_policy_registry']=[{**current,'revision':999,'credential_scope':'other'}]
        self.assertEqual(audit.authority_outcome(state),'verify policy')

    def test_visible_question_and_contract_mutations_rejected(self):
        value=row(joint_state(),'joint_capacity','execute')
        self.assertEqual(audit.replay(value),'execute')
        value['state']['trusted_policy']['rules']=value['state']['trusted_policy']['rules'].replace('equality means execute','equality means request higher capacity')
        with self.assertRaisesRegex(ValueError,'authored rule'):audit.replay(value)
        value=row(joint_state(),'joint_capacity','execute')
        value.update(kind='noul',options=['no','yes'],target=[0.,1.],question="Does the supplied rule establish 'execute' for this case?")
        value['metadata']['proposed_outcome']='execute'
        audit.replay(value)
        value['question']="Does the supplied rule establish 'request higher capacity' for this case?"
        with self.assertRaisesRegex(ValueError,'Visible Noul'):audit.replay(value)
        value['metadata']['proposed_outcome']='request higher capacity'
        with self.assertRaisesRegex(ValueError,'target'):audit.replay(value)


class IndependentDirectoryTests(unittest.TestCase):
    def setUp(self):
        temporal={'temporal_window':audit.CONDITIONS['temporal_window']}
        patched=patch.object(audit,'CONDITIONS',temporal);patched.start();self.addCleanup(patched.stop)
        self.rows=[]
        for split in SPLITS:
            cursor=0
            for index in range(audit.GROUP_COUNTS[split]):
                group='manual/'+split+'/'+str(index);entity='entity-'+group
                start=datetime(2044 if split == 'ood' else 2031,1,1,tzinfo=timezone.utc)+timedelta(days=index)
                side=(index//2+index//4)%2
                cases=[(-1,False,'reject'),(0,False,'accept'),(2,False,'accept'),(4,False,'accept'),
                       (5,False,'reject'),(-1,True,'review'),(5,True,'review'),(-1 if side == 0 else 5,False,'reject')]
                for case,(delta,exception,answer) in enumerate(cases):
                    condition=temporal['temporal_window'][case]
                    instant=start+timedelta(seconds=delta)
                    if case == 7:instant=instant.astimezone(timezone(timedelta(hours=2)))
                    state=dict(case_ref=entity,delivered_at=start.isoformat(),request_received_at=instant.isoformat(),
                        return_window_seconds=4,exception_approved=exception,policy=audit.CONTRACTS['temporal_window'])
                    value=row(state,'temporal_window',answer)
                    value.update(id=group+'/v'+str(case),group_id=group,split=split)
                    if (index+case)%2:
                        truth=(index//2+case//2)%2 == 0
                        proposed=answer if truth else next(label for label in ('accept','reject','review') if label != answer)
                        value.update(kind='noul',options=['no','yes'],target=[float(not truth),float(truth)],
                            question="Does the supplied rule establish '"+proposed+"' for this case?")
                        value['metadata']['proposed_outcome']=proposed
                    else:
                        options=[label for label in ('accept','reject','review') if label != answer]
                        options.insert(cursor%3,answer);cursor+=1
                        value.update(options=options,target=[float(label == answer) for label in options])
                    value['metadata'].update(family='policy',condition=condition,entity_ids=[entity],
                        template_id='manual-temporal/'+('ood' if split == 'ood' else 'id'),
                        provenance=dict(type='synthetic',license='CC0-1.0',generator_version=audit.VERSION,seed=1,
                            group_index=index,variant=case,split_policy='whole_groups',upstream_rows_imported=0))
                    self.rows.append(value)

    def test_complete_hand_authored_groups_and_structural_limits(self):
        result=audit.audit_rows(self.rows)
        self.assertEqual(result['oracle_checked'],448)
        self.assertEqual(result['complete_groups'],56)
        self.assertTrue(result['structural_coverage_limits'])
        self.assertTrue(all(item['split'] in ('calibration','validation') for item in result['structural_coverage_limits']))

    def test_condition_target_and_incomplete_group_rejected(self):
        changed=copy.deepcopy(self.rows)
        changed[0]['metadata']['condition'],changed[1]['metadata']['condition']='at_start','before'
        with self.assertRaisesRegex(ValueError,'Condition label'):audit.audit_rows(changed)
        changed=copy.deepcopy(self.rows)
        old=changed[0]['target'].index(1.);changed[0]['target']=[float(i == (old+1)%3) for i in range(3)]
        with self.assertRaisesRegex(ValueError,'target'):audit.audit_rows(changed)
        with self.assertRaisesRegex(ValueError,'eight-variant'):audit.audit_rows(self.rows[1:])

    def test_directory_receipt_hashes_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            _write_dataset(self.rows,directory,{})
            result=audit.audit_directory(directory)
            self.assertEqual(result['status'],'validated_no_model_run')
            self.assertEqual(result['manifest_sha256'],audit.file_sha256(directory/'manifest.json'))
            path=directory/'test.jsonl';path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'split hash'):audit.audit_directory(directory)

    def test_cli_writes_fresh_receipt_and_sparse_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'candidate';_write_dataset(self.rows,directory,{})
            output=Path(temporary)/'audit.json'
            arguments=['audit','--dataset',str(directory),'--output',str(output)]
            with patch('sys.argv',arguments),patch('sys.stdout',new_callable=io.StringIO) as stdout:
                audit.main()
            self.assertEqual(json.loads(stdout.getvalue()),dict(status='validated_no_model_run',oracle_checked=448,complete_groups=56))
            self.assertEqual(json.loads(output.read_text())['status'],'validated_no_model_run')
            with patch('sys.argv',arguments),self.assertRaises(FileExistsError):audit.main()

    def test_rehashed_wrong_rule_still_fails_independent_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed=copy.deepcopy(self.rows)
            changed[0]['state']['policy']=changed[0]['state']['policy'].replace('including both endpoints','excluding both endpoints')
            _write_dataset(changed,temporary,{})
            with self.assertRaisesRegex(ValueError,'authored rule'):audit.audit_directory(temporary)

    def test_kind_truth_coupled_structure_rejected(self):
        changed=copy.deepcopy(self.rows)
        for value in changed:
            if value['metadata']['condition'] != 'equivalent_outside':continue
            state=value['state'];start=datetime.fromisoformat(state['delivered_at'])
            # Reintroduce the previous parity shortcut while preserving exact gold.
            parity=value['metadata']['provenance']['group_index']%2
            state['request_received_at']=(start+timedelta(seconds=-1 if parity == 0 else 5)).astimezone(timezone(timedelta(hours=2))).isoformat()
        with self.assertRaisesRegex(ValueError,'structural axis'):audit.audit_rows(changed)


if __name__ == '__main__':unittest.main()
