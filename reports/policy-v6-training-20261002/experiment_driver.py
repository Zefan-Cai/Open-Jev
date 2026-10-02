"""Observe the frozen training call, bind its result, then run fixed comparison."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
COMMIT = '1ecbabb77d506d948dc7dd548f1cedb2f3f8f189'
PLAN_SHA = 'c3f44415e7000fa5a7c0653c85519956596d1ff845ee2afed82d9777f5174c82'
DATA = ROOT/'data/policy-training-v6-20261002-r1'
SOURCE = ROOT/'source'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_once(name, value):
    with (ROOT/name).open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def main():
    assert os.getsid(0) == os.getpid(), 'Driver must lead its own owned session'
    stage = json.loads((ROOT/'cpu-stage-receipt.json').read_text())
    plan = json.loads((DATA/'comparison-plan.json').read_text())
    assert stage['status'] == 'cpu_staged_no_cuda_initialization'
    assert stage['source_commit'] == plan['source_commit'] == COMMIT
    assert sha(DATA/'comparison-plan.json') == stage['plan_sha256'] == PLAN_SHA
    assert not (ROOT/'adaptation').exists(), 'Refuse to reuse a training attempt'
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip()
    request_sha = sha(ROOT/'comparison-request.json')
    request = json.loads((ROOT/'comparison-request.json').read_text())
    evaluation_source = Path(request['source_directory'])
    assert evaluation_source == ROOT/'evaluation-source'
    assert subprocess.check_output(['git','-C',str(evaluation_source),'rev-parse','HEAD'],text=True).strip() == request['source_commit']
    assert sha(evaluation_source/'scripts/compare_policy_training_v6.py') == request['comparison_script_sha256']
    command = request['command']
    assert command[:3] == [sys.executable,'-m','scripts.compare_policy_training_v6']
    removed = [key for key in os.environ if key.startswith('JEV_')]
    for key in removed:
        os.environ.pop(key)
    os.environ['JEV_TORCH_DTYPE'] = 'bfloat16'
    sys.path.insert(0, str(SOURCE))
    from jev import train
    from scripts.train_temporal_windows_v5 import directory_sha
    import torch
    from safetensors.torch import load_file
    assert not torch.cuda.is_initialized()
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
    assert {name:importlib.metadata.version(name) for name in stage['runtime']} == stage['runtime']
    original_initialize = train.initialize_model
    original_step = torch.optim.AdamW.step
    observed = []
    first_step = False

    def initialize(args, model_class, identity):
        assert identity is not None
        expected = plan['expected_initial_checkpoint']
        checkpoint = Path(args.initial_checkpoint)
        files = {p.relative_to(checkpoint).as_posix():sha(p) for p in checkpoint.rglob('*') if p.is_file()}
        assert files == expected['files_sha256']
        assert directory_sha(checkpoint) == expected['directory_sha256']
        model = original_initialize(args, model_class, identity)
        saved = load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
        loaded = {}
        inventory = []
        counts = {'lora_A':0,'lora_B':0,'head':0,'base':0}
        for name, parameter in model.named_parameters():
            group = 'head' if name.startswith('head.') else 'lora_A' if '.lora_A.' in name else 'lora_B' if '.lora_B.' in name else 'base'
            counts[group] += parameter.numel()
            assert parameter.requires_grad == (group != 'base'), name
            assert parameter.dtype == (torch.bfloat16 if group == 'base' else torch.float32), name
            if group in ('lora_A','lora_B'):
                key = name.removeprefix('backbone.').replace('.default.','.')
                loaded[key] = parameter.detach().cpu()
                inventory.append({'name':name,'numel':parameter.numel(),'dtype':str(parameter.dtype)})
        assert all(counts.values())
        assert loaded.keys() == saved.keys(), 'Actual LoRA key set differs from saved weights'
        for name, value in loaded.items():
            assert torch.equal(value, saved[name].float()), name
        saved_head = torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=True)
        assert model.head.state_dict().keys() == saved_head.keys()
        for name, value in model.head.state_dict().items():
            assert torch.equal(value.detach().cpu(), saved_head[name].float()), name
        active = list(model.backbone.active_adapters)
        assert active == ['default']
        layers = [layer for layer in model.backbone.modules() if hasattr(layer,'lora_A')]
        assert layers and all(not layer.merged and not layer.disable_adapters for layer in layers)
        write_once('initial-load-verification.json', dict(status='passed', checkpoint_files_sha256=files,
            checkpoint_directory_sha256=directory_sha(checkpoint), loaded_lora_and_head_exact=True,
            parameter_numel=counts,lora_inventory=inventory,active_adapters=active,
            merged=False,disable_adapters=False,base_frozen=True,
            removed_environment_names=removed,training_source_commit=COMMIT,driver_sha256=sha(__file__)))
        observed.append(model)
        return model

    def step(optimizer, *args, **kwargs):
        nonlocal first_step
        if not first_step:
            gradients = {'lora_A':[],'lora_B':[],'head':[]}
            for name, parameter in observed[0].named_parameters():
                group = 'head' if name.startswith('head.') else 'lora_A' if '.lora_A.' in name else 'lora_B' if '.lora_B.' in name else 'base'
                if group == 'base':
                    assert parameter.grad is None, name
                    continue
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all().item(), name
                gradients[group].append(float(parameter.grad.detach().abs().max().item()))
            assert all(gradients.values()) and max(gradients['lora_A']) > 0 and max(gradients['lora_B']) > 0
            write_once('first-step-gradient-verification.json', dict(status='passed',finite=True,
                lora_A_nonzero=True,lora_B_nonzero=True,base_gradients_absent=True,
                gradient_parameter_counts={k:len(v) for k,v in gradients.items()},
                gradient_max_abs={k:max(v) for k,v in gradients.items()},
                checked_after_clipping_before_first_optimizer_update=True))
            first_step = True
            observed.clear()
        return original_step(optimizer, *args, **kwargs)

    train.initialize_model = initialize
    torch.optim.AdamW.step = step
    started = time.time()
    os.chdir(ROOT)
    sys.argv = ['jev.train', *plan['training_argv'][3:]]
    try:
        train.main()
    finally:
        train.initialize_model = original_initialize
        torch.optim.AdamW.step = original_step
    assert first_step
    observed.clear()
    gc.collect()
    torch.cuda.empty_cache()
    run = ROOT/'adaptation'
    summary = json.loads((run/'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['steps'] == 692 and summary['trained_rows_consumed'] == 2768
    artifacts = {name:sha(run/name) for name in ('run.json','summary.json','training.jsonl','calibration.jsonl')}
    checkpoint = run/'checkpoint'
    files = {p.relative_to(checkpoint).as_posix():sha(p) for p in checkpoint.rglob('*') if p.is_file()}
    write_once('completion-receipt.json', dict(status='complete',source_commit=COMMIT,
        artifacts_sha256=artifacts,checkpoint={'files_sha256':files,'directory_sha256':directory_sha(checkpoint)},
        driver_sha256=sha(__file__),cpu_stage_receipt_sha256=sha(ROOT/'cpu-stage-receipt.json'),
        initial_load_verification_sha256=sha(ROOT/'initial-load-verification.json'),
        first_step_gradient_verification_sha256=sha(ROOT/'first-step-gradient-verification.json'),
        elapsed_seconds=time.time()-started,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
    assert sha(ROOT/'comparison-request.json') == request_sha
    assert subprocess.check_output(['git','-C',str(evaluation_source),'rev-parse','HEAD'],text=True).strip() == request['source_commit']
    assert sha(evaluation_source/'scripts/compare_policy_training_v6.py') == request['comparison_script_sha256']
    environment = {**os.environ,'PYTHONPATH':str(ROOT/'overlay')+':'+str(evaluation_source)}
    with (ROOT/'comparison.log').open('x') as log:
        result = subprocess.run(command,cwd=evaluation_source,env=environment,stdout=log,stderr=subprocess.STDOUT)
    write_once('experiment-completion.json', dict(training_status='complete',comparison_returncode=result.returncode,
        training_completion_receipt_sha256=sha(ROOT/'completion-receipt.json'),
        comparison_request_sha256=sha(ROOT/'comparison-request.json')))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
