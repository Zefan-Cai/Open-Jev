"""CPU contracts for the fixed final comparison; no Torch/CUDA dependency."""
from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import io
import json
from pathlib import Path
import math
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev.data import SPLITS, _write_dataset
from jev.frontier_controls_v4 import records as v4_records
from jev.policy_controls_v6 import records as v6_records
from jev.temporal_windows_v5 import records as v5_records
from jev.train import _file_sha256, _json_sha256, read_rows
from scripts import compare_policy_training_v6 as comparison


def journal(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def prediction(row, action=None):
    action = action or row['options'][row['target'].index(1.)]
    return {**{key:copy.deepcopy(row[key]) for key in ('id','group_id','source','kind','options','target')},
            'row_sha256':_json_sha256(row),'status':'complete',
            'logits':[3. if label == action else .5 for label in row['options']]}


class ComparisonMetricsTests(unittest.TestCase):
    def test_noul_inclusive_thresholds_nextafter_and_zero_accepted(self):
        probabilities = [math.nextafter(.2,0.),.2,math.nextafter(.2,1.),
                         math.nextafter(.8,0.),.8,math.nextafter(.8,1.)]
        rows = [{'kind':'noul','target':[float(p < .5),float(p >= .5)],'logits':[p]} for p in probabilities]
        with patch.object(comparison,'softmax',side_effect=lambda logits,t:[1-logits[0],logits[0]]):
            result = comparison.noul_metrics(rows,1.)
            self.assertEqual((result['n'],result['accepted'],result['correct'],result['abstentions']),(6,4,4,2))
            self.assertEqual(result['accuracy'],4/6)
            self.assertEqual(result['accepted_errors'],0)
            result = comparison.noul_metrics(rows[2:4],1.)
            self.assertEqual((result['accuracy'],result['coverage'],result['accepted_errors']),(0.,0.,0))
            self.assertIsNone(result['accepted_accuracy'])
            self.assertIsNone(result['error_among_accepted'])
            rows[0]['target'] = [0.,1.]
            self.assertEqual(comparison.noul_metrics(rows,1.)['accepted_errors'],1)

    def test_four_temperature_cells_preserve_argmax_and_change_probability_metrics(self):
        base = [{'kind':'choice','target':[1.,0.],'logits':[2.,0.]},
                {'kind':'noul','target':[0.,1.],'logits':[0.,1.]}]
        adapted = copy.deepcopy(base)
        adapted[0]['logits'] = [.2,0.]
        adapted[1]['logits'] = [0.,3.]
        cells = [comparison.metrics(rows,t) for rows in (base,adapted) for t in (1.,3.)]
        self.assertEqual([cell['accuracy'] for cell in cells],[1.]*4)
        self.assertNotEqual(cells[0]['nll'],cells[1]['nll'])
        self.assertNotEqual(cells[2]['brier'],cells[3]['brier'])
        self.assertNotEqual(cells[2]['noul_thresholds_0.2_0.8']['coverage'],cells[3]['noul_thresholds_0.2_0.8']['coverage'])

    def test_injection_options_align_by_text_and_consistent_wrong_is_separate(self):
        candidates = [r for r in v6_records() if r['split'] == 'test'
                      and r['metadata']['scenario_family'] == 'untrusted_policy_conflict']
        group = candidates[0]['group_id']
        rows = [r for r in candidates if r['group_id'] == group]
        rows[1]['options'].reverse();rows[1]['target'].reverse()
        result = comparison.injection_pairs([prediction(r) for r in rows],rows,1.)
        self.assertEqual((result['pairs'],result['same_action'],result['both_correct']),(1,1,1))
        self.assertAlmostEqual(result['mean_max_label_probability_change'],0.,delta=1e-15)
        wrong = next(label for label in rows[0]['options'] if label != rows[0]['options'][rows[0]['target'].index(1.)])
        values = [prediction(r,wrong) if i < 2 else prediction(r) for i,r in enumerate(rows)]
        result = comparison.injection_pairs(values,rows,1.)
        self.assertEqual((result['same_action'],result['both_correct'],result['same_action_both_wrong']),(1,0,1))
        values[0] = prediction(rows[0])
        self.assertEqual(comparison.injection_pairs(values,rows,1.)['benign_correct_injected_wrong'],1)
        rows[1]['state']['request']['amount_cents'] += 1
        with self.assertRaisesRegex(ValueError,'verified facts'):
            comparison.injection_pairs(values,rows,1.)

    def test_semantic_temporal_and_joint_authority_subgroups(self):
        for split in ('test','ood'):
            rows = [r for r in v5_records() if r['split'] == split]
            counts = Counter(comparison.labels(r)['temporal_boundary'] for r in rows)
            self.assertEqual(counts,dict(before_delivery=4,at_delivery=4,inside_window=8,at_deadline=4,after_deadline=4,exception=8))
            self.assertEqual(Counter(r['kind'] for r in rows if comparison.labels(r)['rejection'] == 'outside_window'),
                             dict(choice=6,noul=2))
            rows = [r for r in v6_records() if r['split'] == split and r['metadata']['scenario_family'] == 'scoped_joint_approval']
            self.assertEqual(Counter(comparison.labels(r)['amount_boundary'] for r in rows),
                             dict(equal_capacity=3,one_cent_above=3,authority_blocked=6))
            self.assertEqual(Counter(comparison.labels(r)['rejection'] for r in rows),
                             dict(not_rejected=6,revoked=3,missing_scope_consent=3))
            first = copy.deepcopy(rows[0]);first['state']['signed_events'].reverse()
            self.assertEqual(comparison.labels(first),comparison.labels(rows[0]))

    def test_prediction_binding_and_paired_transitions(self):
        row = next(v6_records());value = prediction(row)
        comparison.verify_prediction_rows([value],[row],full=True)
        for key,bad in (('id','wrong'),('options',list(reversed(value['options']))),('row_sha256','0'*64),
                        ('logits',[0.]),('logits',[math.nan]*len(row['options'])),('status','failed')):
            changed = copy.deepcopy(value);changed[key] = bad
            with self.subTest(key=key),self.assertRaises(ValueError):
                comparison.verify_prediction_rows([changed],[row],full=True)
        changed = copy.deepcopy(value);del changed['row_sha256']
        with self.assertRaisesRegex(ValueError,'incomplete'):
            comparison.verify_prediction_rows([changed],[row],full=True)
        wrong = prediction(row,next(x for x,t in zip(row['options'],row['target']) if not t))
        transitions = comparison.paired_decisions([value,wrong],[wrong,value])
        self.assertEqual((transitions['argmax_changed'],transitions['correct_to_incorrect'],transitions['incorrect_to_correct']),(2,1,1))

    def test_frozen_source_and_committed_evaluation_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ['scripts/compare_policy_training_v6.py',*[f'jev/{n}.py' for n in
                     ('train','model','api','metrics','data','frontier_controls_v4','temporal_windows_v5','policy_controls_v6')]]
            for name in names:
                path = root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixed')
            with patch.object(comparison,'ROOT',root),patch.object(comparison,'source_checkout_commit',return_value='eval'), \
                 patch.object(comparison.subprocess,'check_output',return_value=b'fixed') as git_show:
                self.assertEqual(comparison.source_identity('eval','frozen')['commit'],'eval')
                git_show.side_effect = lambda cmd,**kw:b'changed' if cmd[-1] == 'frozen:jev/train.py' else b'fixed'
                with self.assertRaisesRegex(ValueError,'frozen training'):
                    comparison.source_identity('eval','frozen')
                git_show.side_effect = None
                (root/names[0]).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Uncommitted'):
                    comparison.source_identity('eval','frozen')


class ComparisonPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset,self.training = self.root/'data',self.root/'adaptation'
        self.training.mkdir()
        self.source_rows = {}
        for source,records in (('v4',v4_records(groups_per_family=1)),('v5',v5_records()),('v6',v6_records())):
            source_rows = list(records)
            self.source_rows[source] = {}
            for split in SPLITS:
                group = next(r['group_id'] for r in source_rows if r['split'] == split)
                self.source_rows[source][split] = [r for r in source_rows if r['group_id'] == group]
        main = [r for split in SPLITS for source in (('v6',) if split in ('test','ood') else self.source_rows)
                for r in self.source_rows[source][split]]
        self.manifest = _write_dataset(main,self.dataset,{})
        heldout,regression = {},{}
        for source in self.source_rows:
            for split in ('calibration','validation'):
                path = self.dataset/'heldout'/source/(split+'.jsonl');journal(path,self.source_rows[source][split])
                heldout[path.relative_to(self.dataset).as_posix()] = _file_sha256(path)
            if source != 'v6':
                regression[source] = {'selected_rows_sha256':{}}
                for split in ('test','ood'):
                    rows = self.source_rows[source][split]
                    path = self.dataset/'observed-regression'/source/(split+'.jsonl');journal(path,rows)
                    heldout[path.relative_to(self.dataset).as_posix()] = _file_sha256(path)
                    regression[source]['selected_rows_sha256'][split] = _json_sha256(rows)
        self.settings = dict(steps=3,accumulation=4,train_rows=12,calibration_rows=12,eval_rows=4,
            max_length=4096,lora_rank=8,seed=20261002,checkpoint_every=0,lr=2e-5,head_lr=5e-5,
            brier_weight=.1,training_sampling='shuffled')
        self.plan = dict(source_commit='a'*40,settings=self.settings,
            data_manifest_sha256=_file_sha256(self.dataset/'manifest.json'),heldout_files_sha256=heldout,
            observed_regression_locks=regression,
            training_argv=['python','-m','jev.train','--model','fixture/base','--revision','b'*40])
        self.plan_path = self.root/'plan.json'
        self.args = SimpleNamespace(dataset=str(self.dataset),training_run=str(self.training),
            released_checkpoint=str(self.root/'released'),output=str(self.root/'comparison'),
            completion_receipt=str(self.root/'completion-receipt.json'),plan=str(self.plan_path),
            expected_commit='eval',device='cuda:0')
        calibration = [prediction(r) for r in read_rows(self.dataset/'calibration.jsonl',12,20261002,balanced=True)]
        for index,value in enumerate(calibration):
            if index % 3 == 0:
                value['logits'] = list(reversed(value['logits']))
        self.temperature = comparison.fit_temperature([r['logits'] for r in calibration],[r['target'] for r in calibration])
        journal(self.training/'calibration.jsonl',calibration)
        ids = [r['id'] for r in calibration]
        self.checkpoint(Path(self.args.released_checkpoint),1.5)
        self.checkpoint(self.training/'checkpoint',self.temperature,dict(split='calibration',n=12,
            ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
        released = comparison.checkpoint_identity(self.args.released_checkpoint,self.args.output,self.plan)
        self.plan['expected_initial_checkpoint'] = dict(files_sha256=released['files_sha256'],
            directory_sha256=comparison.directory_sha(self.args.released_checkpoint))
        self.meta = {**self.settings,'model':'fixture/base','revision':'b'*40,'commit':self.plan['source_commit'],
            'resume_training':None,'baseline_initialization':'inference_checkpoint',
            'data_sha256':{Path(n).stem:s for n,s in self.manifest['files_sha256'].items()},
            'initial_checkpoint_identity':released,'calibration_ids':ids,
            'evaluation_ids':[r['id'] for r in read_rows(self.dataset/'test.jsonl',4,20261002,balanced=True)],
            'ood_ids':[r['id'] for r in read_rows(self.dataset/'ood.jsonl',4,20261002,balanced=True)]}
        self.summary = dict(status='complete',model='fixture/base',steps=3,trained_rows_consumed=12,
                            checkpoint_reload_max_error=0.,temperature=self.temperature)
        comparison.write_json(self.training/'run.json',self.meta)
        comparison.write_json(self.training/'summary.json',self.summary)
        journal(self.training/'training.jsonl',[dict(step=i,loss=.5,gradient_norm=.1) for i in range(1,4)])
        self.refresh_receipt()
        self.refresh_plan()
        patched = patch.object(comparison,'EVAL_COUNTS',{s+'_'+split:4 for s in self.source_rows for split in ('test','ood')})
        patched.start();self.addCleanup(patched.stop)

    def checkpoint(self,path,temperature,extra=None):
        (path/'adapter').mkdir(parents=True)
        comparison.write_json(path/'model.json',dict(method='independent_candidate_lora_nll_brier',
            model_id='fixture/base',revision='b'*40,lora_rank=8,max_length=4096))
        comparison.write_json(path/'adapter/adapter_config.json',dict(peft_type='LORA',r=8))
        (path/'adapter/adapter_model.safetensors').write_bytes(b'fake weights '+str(temperature).encode())
        (path/'head.pt').write_bytes(b'fake head')
        comparison.write_json(path/'temperature.json',dict(temperature=temperature,**(extra or {})))

    def refresh_receipt(self):
        checkpoint = self.training/'checkpoint'
        comparison.write_json(self.args.completion_receipt,dict(status='complete',source_commit=self.plan['source_commit'],
            artifacts_sha256={n:_file_sha256(self.training/n) for n in ('run.json','summary.json','training.jsonl','calibration.jsonl')},
            checkpoint={'directory_sha256':comparison.directory_sha(checkpoint),
                'files_sha256':{p.relative_to(checkpoint).as_posix():_file_sha256(p) for p in checkpoint.rglob('*') if p.is_file()}},
            exit_code=0))

    def refresh_plan(self):
        comparison.write_json(self.plan_path,self.plan)
        patched = patch.object(comparison,'PLAN_SHA256',_file_sha256(self.plan_path));patched.start();self.addCleanup(patched.stop)

    def test_valid_preflight_locks_complete_receipt_and_all_six_sets(self):
        plan,rows,released,adapted,inputs = comparison.prepare(self.args)
        self.assertEqual({n:len(r) for n,r in rows.items()},comparison.EVAL_COUNTS)
        self.assertEqual(released['temperature'],1.5)
        self.assertEqual(adapted['temperature'],self.temperature)
        self.assertEqual(inputs['completion_receipt']['exit_code'],0)
        self.assertIn(self.args.completion_receipt,inputs['files_sha256'])
        self.assertFalse(Path(self.args.output).exists())

    def test_plan_and_data_byte_tampering_rejected(self):
        self.plan_path.write_text(self.plan_path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'public plan'):
            comparison.prepare(self.args)
        self.refresh_plan()
        path = self.dataset/'test.jsonl';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Frozen input hash'):
            comparison.prepare(self.args)

    def test_source_settings_completion_and_reload_error_rejected(self):
        for key,value in (('commit','c'*40),('lr',1e-3),('resume_training','snapshot')):
            changed = {**self.meta,key:value};comparison.write_json(self.training/'run.json',changed)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Training completion'):
                comparison.prepare(self.args)
        comparison.write_json(self.training/'run.json',self.meta)
        comparison.write_json(self.training/'summary.json',{**self.summary,'checkpoint_reload_max_error':.06})
        with self.assertRaisesRegex(ValueError,'Training completion'):
            comparison.prepare(self.args)

    def test_swapped_final_weights_or_modified_completion_receipt_rejected(self):
        path = self.training/'checkpoint/head.pt';path.write_bytes(b'replaced final head')
        with self.assertRaisesRegex(ValueError,'completion receipt'):
            comparison.prepare(self.args)
        path.write_bytes(b'fake head')
        receipt = json.loads(Path(self.args.completion_receipt).read_text());receipt['source_commit'] = 'wrong'
        comparison.write_json(self.args.completion_receipt,receipt)
        with self.assertRaisesRegex(ValueError,'completion receipt'):
            comparison.prepare(self.args)

    def test_calibration_order_temperature_and_split_cannot_be_changed(self):
        path = self.training/'calibration.jsonl';values = comparison.read(path);journal(path,list(reversed(values)))
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError,'identity differs'):
            comparison.prepare(self.args)
        journal(path,values)
        path = self.training/'checkpoint/temperature.json';value = json.loads(path.read_text());value['split'] = 'test'
        comparison.write_json(path,value);self.refresh_receipt()
        with self.assertRaisesRegex(ValueError,'Calibration logits'):
            comparison.prepare(self.args)
        value.update(split='calibration',temperature=self.temperature+1.)
        comparison.write_json(path,value)
        comparison.write_json(self.training/'summary.json',{**self.summary,'temperature':value['temperature']})
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError,'Calibration logits'):
            comparison.prepare(self.args)

    def test_regression_order_and_output_containment_rejected(self):
        path = self.dataset/'observed-regression/v4/test.jsonl';journal(path,list(reversed(comparison.read(path))))
        self.plan['heldout_files_sha256']['observed-regression/v4/test.jsonl'] = _file_sha256(path);self.refresh_plan()
        with self.assertRaisesRegex(ValueError,'regression order'):
            comparison.prepare(self.args)
        for directory in (self.dataset,self.training):
            self.args.output = str(directory/'comparison')
            with self.assertRaisesRegex(ValueError,'outside the frozen'):
                comparison.prepare(self.args)

    def test_sequential_model_lifecycle_lock_and_four_cell_output_without_gpu(self):
        events = []
        fixture = self

        class Tensor:
            def __init__(self,values):self.values = values
            def float(self):return self
            def cpu(self):return self
            def tolist(self):return self.values

        class Model:
            live = 0
            tamper = False
            @classmethod
            def load(cls,path,device):
                fixture.assertEqual(cls.live,0)
                fixture.assertTrue((Path(fixture.args.output)/'comparison.lock.json').exists())
                events.append('load');cls.live += 1
                if cls.tamper and Path(path) == fixture.training/'checkpoint':
                    (Path(path)/'added-during-inference').write_text('changed')
                return cls()
            def eval(self):pass
            def __call__(self,rows):return [Tensor(prediction(rows[0])['logits'])]
            def __del__(self):Model.live -= 1;events.append('delete')

        torch = SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(is_available=lambda:True,
            synchronize=lambda device:None,empty_cache=lambda:events.append('empty_cache')))
        with patch.dict(sys.modules,{'torch':torch,'jev.model':SimpleNamespace(DecisionModel=Model)}), \
             patch.object(comparison,'source_identity',return_value={'commit':'eval'}), \
             patch.object(comparison,'runtime_identity',lambda *args:{'same_runtime':True}),patch('sys.stdout',new_callable=io.StringIO):
            comparison.run(self.args)
            self.assertEqual(events,['load','delete','empty_cache','load','delete','empty_cache'])
            result = json.loads((Path(self.args.output)/'summary.json').read_text())
            Model.tamper = True;self.args.output = str(self.root/'tampered-comparison')
            with self.assertRaisesRegex(ValueError,'Source/checkpoint changed'):
                comparison.run(self.args)
            self.assertFalse((Path(self.args.output)/'summary.json').exists())
            self.assertEqual(Model.live,0)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(result['journal_files_sha256']),12)
        self.assertTrue(all(len(cells) == 4 for cells in result['metrics'].values()))

    def test_nonfinite_inference_writes_failed_record_and_raises(self):
        row = self.source_rows['v6']['test'][0]
        tensor = SimpleNamespace(float=lambda:None)
        tensor.float = lambda:tensor;tensor.cpu = lambda:tensor;tensor.tolist = lambda:[math.nan]*len(row['options'])
        class Model:
            def eval(self):pass
            def __call__(self,rows):return [tensor]
        torch = SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(synchronize=lambda device:None))
        path = self.root/'failed.jsonl'
        with self.assertRaises(ValueError):
            comparison.predict(Model(),[row],path,1.,'cuda:0',torch)
        values = comparison.read(path)
        self.assertEqual(values[0]['status'],'failed')
        self.assertNotIn('logits',values[0]);self.assertNotIn('probabilities',values[0])

    def test_first_cuda_synchronize_failure_is_journaled(self):
        row = self.source_rows['v6']['test'][0]
        def fail(device):raise RuntimeError('synchronize failed')
        torch = SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(synchronize=fail))
        path = self.root/'sync-failed.jsonl'
        with self.assertRaisesRegex(RuntimeError,'synchronize failed'):
            comparison.predict(SimpleNamespace(eval=lambda:None),[row],path,1.,'cuda:0',torch)
        value = comparison.read(path)[0]
        self.assertEqual(value['status'],'failed')
        self.assertEqual(value['id'],row['id'])


if __name__ == '__main__':
    unittest.main()
