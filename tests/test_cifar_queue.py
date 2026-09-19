import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from drift_cil.data import DataBundle
from drift_cil.runner import run
from run_pilot import run_queue

ROOT=Path(__file__).resolve().parents[1]


def test_cifar_transform_and_debug_split_keep_original_indices(monkeypatch):
    from torchvision import datasets
    class FakeCIFAR:
        def __init__(self,root,train,download):
            self.classes=[f"class_{i}" for i in range(100)]
            self.targets=np.repeat(np.arange(100),5 if train else 2).tolist()
            self.data=np.zeros((len(self.targets),32,32,3),dtype=np.uint8)
    monkeypatch.setattr(datasets,"CIFAR100",FakeCIFAR)
    cfg=yaml.safe_load((ROOT/"configs/cifar100.yaml").read_text())
    cfg.update(validation_per_class=1,debug_train_per_class=1,debug_validation_per_class=1)
    data=DataBundle(cfg)
    assert data.debug_subset
    assert len(data.train_indices)==len(data.val_indices)==100
    assert not set(data.train_indices)&set(data.val_indices)
    assert len(data.pool(0))==10
    x,y=data.batch(data.pool(0)[:2],torch.device("cpu"))
    assert x.shape==(2,3,224,224)
    expected=-torch.tensor([.48145466,.4578275,.40821073])/torch.tensor([.26862954,.26130258,.27577711])
    assert torch.allclose(x[0,:,0,0],expected)
    assert set(y.tolist())<=set(data.task_classes[0])


def test_resume_rejects_momentum_change_previously_unchecked(tmp_path):
    cfg=yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    run(cfg,tmp_path/"run","drift",task_limit=1,print_every=10000)
    with pytest.raises(ValueError,match="beta"):
        run(dict(cfg,beta=.9),tmp_path/"run","drift",resume=tmp_path/"run/last.pt")


def test_queue_skip_extend_and_config_guard(tmp_path):
    config=tmp_path/"config.yaml"
    cfg=yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    config.write_text(yaml.safe_dump(cfg))
    output=tmp_path/"queue"
    assert run_queue(config,output,["drift"],[1993],1)==0
    log=output/"drift_s1993/console.log"
    before=log.read_bytes()
    assert run_queue(config,output,["drift"],[1993],1)==0
    assert log.read_bytes()==before
    assert run_queue(config,output,["drift"],[1993],2)==0
    summary=json.loads((output/"drift_s1993/summary.json").read_text())
    assert summary["completed_tasks"]==2
    assert (output/"comparison.csv").exists()
    cfg["lr"]*=2;config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match="Config changed"):
        run_queue(config,output,["drift"],[1993],2)
