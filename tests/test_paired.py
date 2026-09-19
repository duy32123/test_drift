import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from drift_cil.runner import file_hash, run
from run_paired import run_pair

ROOT = Path(__file__).resolve().parents[1]


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def test_fork_restores_entire_task_a_state_and_guards_method(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    run(cfg, tmp_path / "common", "drift", task_limit=1)
    source = tmp_path / "common/task_01.pt"
    saved = torch.load(source, weights_only=False)
    for method in ("drift", "drift_scalar"):
        out = tmp_path / method
        run(cfg, out, method, task_limit=1, fork_from=source)
        fork = torch.load(out / "last.pt", weights_only=False)
        for key in ("adapters", "momentum", "memory", "torch_rng", "numpy_rng",
                    "python_rng", "cuda_rng", "label_rng", "stage_metrics", "update"):
            assert_same(saved[key], fork[key])
        assert fork["fork_provenance"]["checkpoint_sha256"] == file_hash(source)
    with pytest.raises(ValueError, match="different method"):
        run(cfg, tmp_path / "bad_resume", "drift_scalar", resume=source)
    with pytest.raises(ValueError, match="only supports"):
        run(cfg, tmp_path / "bad_method", "replay", fork_from=source)
    with pytest.raises(ValueError, match="config differs"):
        run(dict(cfg, delta=0.5), tmp_path / "bad_config", "drift_scalar", fork_from=source)
    saved.pop("code_sha256_lf")
    torch.save(saved, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="source hash"):
        run(cfg, tmp_path / "legacy", "drift", fork_from=tmp_path / "legacy.pt")


def test_paired_pipeline_records_common_parent_and_verifiable_artifacts(tmp_path):
    out = tmp_path / "paired"
    reports = run_pair(ROOT / "configs/smoke.yaml", out)
    assert len(reports) == 2
    manifests = [json.loads((out / m / "manifest.json").read_text())
                 for m in ("drift", "drift_scalar")]
    assert manifests[0]["fork_provenance"] == manifests[1]["fork_provenance"]
    summaries = [json.loads((out / m / "summary.json").read_text())
                 for m in ("drift", "drift_scalar")]
    assert summaries[0]["stages"][0] == summaries[1]["stages"][0]
    for report in reports:
        assert report["tolerance"]["matched"]
        assert report["gated_updates"] > 0
        assert set(report["per_task"]) == {"1"}
    for name, digest in json.loads((out / "checksums.json").read_text()).items():
        assert file_hash(out / name) == digest
    with pytest.raises(FileExistsError):
        run_pair(ROOT / "configs/smoke.yaml", out)
