import json
from pathlib import Path
import pytest
import torch
import yaml
from drift_cil.runner import run
from drift_cil.data import DataBundle
from drift_cil.memory import Memory
from drift_cil.model import TinyClassifier

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method",["adamw","muon","scalar","replay","drift","drift_scalar"])
def test_three_task_offline_training(tmp_path,method):
    cfg=yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    result=run(cfg,tmp_path/method,method,print_every=10000)
    assert result["completed_tasks"]==3
    assert result["synthetic"] is True
    assert (tmp_path/method/"last.pt").exists()
    if method in ("scalar","drift"):
        records=[json.loads(x) for x in (tmp_path/method/"steps.jsonl").read_text().splitlines()]
        assert any(r.get("memory_size",0)>0 for r in records)
        for record in records:
            if record.get("within_budget"):
                assert record["memory_after"]<=record["budget_ceiling"]+1.1e-6


def test_task_boundary_resume_matches_uninterrupted(tmp_path):
    cfg=yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    run(cfg,tmp_path/"full","drift",print_every=10000)
    run(cfg,tmp_path/"split","drift",task_limit=1,print_every=10000)
    run(cfg,tmp_path/"split","drift",resume=tmp_path/"split/last.pt",print_every=10000)
    a=torch.load(tmp_path/"full/last.pt",weights_only=False)
    b=torch.load(tmp_path/"split/last.pt",weights_only=False)
    for name in a["adapters"]:
        assert torch.equal(a["adapters"][name],b["adapters"][name]),name
    assert a["stage_metrics"]==b["stage_metrics"]


def test_memory_anchors_survive_class_expansion_and_old_pool_is_not_reloaded():
    cfg=yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    data=DataBundle(cfg);model=TinyClassifier(6,3)
    memory=Memory(12,2)
    memory.update(model,data,0,torch.device("cpu"),4)
    previous={e["index"]:e["anchor"] for e in memory.entries}
    old_classes=set(data.task_classes[0])
    memory.update(model,data,1,torch.device("cpu"),4)
    for e in memory.entries:
        if e["class"] in old_classes:
            assert e["index"] in previous
            assert e["anchor"]==previous[e["index"]]
    assert len(memory)<=12
    assert not set(memory.indices())&set(data.val_indices.tolist())


def test_huggingface_clip_adapter_path_without_download(monkeypatch):
    transformers=pytest.importorskip("transformers")
    from drift_cil.model import CLIPClassifier
    config=transformers.CLIPConfig(
        text_config={"vocab_size":16,"hidden_size":32,"intermediate_size":48,
                     "num_hidden_layers":1,"num_attention_heads":4,"max_position_embeddings":8},
        vision_config={"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
                       "num_attention_heads":4,"image_size":16,"patch_size":4},projection_dim=16)
    source=transformers.CLIPModel(config)
    monkeypatch.setattr(transformers.CLIPModel,"from_pretrained",lambda *a,**kw:source)
    class Tokenizer:
        def __call__(self,texts,**kw):
            return {"input_ids":torch.tensor([[0,3,2]]*len(texts)),
                    "attention_mask":torch.ones(len(texts),3,dtype=torch.long)}
    monkeypatch.setattr(transformers.AutoTokenizer,"from_pretrained",lambda *a,**kw:Tokenizer())
    model=CLIPClassifier("fixture","fixture",["a","b","c"],torch.device("cpu"),3,3,"{}",True)
    x=torch.randn(2,3,16,16)
    model.train();loss=model(x).square().mean();loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    assert len(params)==24
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    model.eval()
    gg=torch.autograd.grad(model(x).square().mean(),params,allow_unused=True)
    assert len(gg)==len(params)
