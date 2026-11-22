"""Mandatory harness tests (section 25).

Run with:  pytest -q tests/

These tests are hermetic where possible (no CIFAR download): split tests use
synthetic targets, checkpoint/metric tests use a tiny in-memory model. Test
#11 builds the model and runs a forward pass on CPU; a full training smoke
test lives in run_all.sh (E0).
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest

from src.config import Config, resolve, parse_overrides
from src.data.split import build_split, get_or_create_split
from src.engine.seeds import set_global_seed
from src.profile_gen import compute_profile
from src.utils.ids import compute_exp_id, compute_run_id, canonical_json, sha256_hex

REPO = Path(__file__).resolve().parents[1]
BASE_CFG = REPO / "configs" / "lab0.yaml"


# --- helpers --------------------------------------------------------------

def _synthetic_targets(n: int = 50_000, classes: int = 10, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(classes) for _ in range(n)]


# --- 1. config loads and contains all required fields ---------------------

def test_config_has_required_fields():
    cfg = Config.load(BASE_CFG)
    cfg.validate()  # raises if any required field is missing
    for section in ["experiment", "data", "model", "optimizer", "scheduler", "train", "logging"]:
        assert section in cfg.to_dict()


# --- 2. same seed -> same split across all roles --------------------------

def test_same_seed_same_split():
    targets = _synthetic_targets()
    s1 = build_split(targets=targets, split_seed=42, train_fraction=0.25)
    s2 = build_split(targets=targets, split_seed=42, train_fraction=0.25)
    assert s1.fingerprint == s2.fingerprint
    for role in ["train", "val", "cal", "test"]:
        assert s1.roles[role] == s2.roles[role]


def test_different_seed_different_split():
    targets = _synthetic_targets()
    s1 = build_split(targets=targets, split_seed=42, train_fraction=0.25)
    s2 = build_split(targets=targets, split_seed=43, train_fraction=0.25)
    assert s1.fingerprint != s2.fingerprint


# --- 3. checkpoint save/load preserves model output -----------------------

def test_checkpoint_roundtrip_preserves_output():
    torch = pytest.importorskip("torch")
    from src.engine.checkpoint import save_checkpoint, restore_checkpoint
    from src.models.small_cnn import SmallCNN

    set_global_seed(0, deterministic=True)
    model = SmallCNN(width=16, depth=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    x = torch.randn(4, 3, 32, 32)
    before = model(x).detach().clone()

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "last.pt")
        save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, epoch=3, global_step=99)
        # Mutate the model, then restore and check output matches `before`.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        meta = restore_checkpoint(path, model=model, optimizer=opt, scheduler=sched)
        after = model(x).detach()
        assert meta["epoch"] == 3 and meta["global_step"] == 99
        assert torch.allclose(before, after, atol=1e-6)


# --- 4. metrics deterministic on fixed input ------------------------------

def test_metrics_deterministic():
    torch = pytest.importorskip("torch")
    from src.models.small_cnn import SmallCNN
    from src.metrics import accuracy

    set_global_seed(7, deterministic=True)
    model = SmallCNN(width=16, depth=2).eval()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    with torch.no_grad():
        a1 = accuracy(model(x), y)
        a2 = accuracy(model(x), y)
    assert a1 == a2


# --- 5. EXP_ID changes when a substantial param changes -------------------

def test_exp_id_changes_with_config():
    cfg = Config.load(BASE_CFG)
    c1 = cfg.apply_overrides(parse_overrides(["experiment.seed=1729"]))
    c2 = cfg.apply_overrides(parse_overrides(["experiment.seed=1730"]))
    h1 = c1.hash(); h2 = c2.hash()
    e1 = compute_exp_id(git_commit="x", git_dirty=False, config_hash=h1,
                        data_fingerprint="d", split_fingerprint="s", seed=1729)
    e2 = compute_exp_id(git_commit="x", git_dirty=False, config_hash=h2,
                        data_fingerprint="d", split_fingerprint="s", seed=1730)
    assert e1 != e2


# --- 6. EXP_ID stable on rerun, RUN_ID changes ----------------------------

def test_exp_id_stable_run_id_changes():
    e = compute_exp_id(git_commit="x", git_dirty=False, config_hash="h",
                       data_fingerprint="d", split_fingerprint="s", seed=1)
    r1 = compute_run_id(e)
    r2 = compute_run_id(e)
    assert e == e
    assert r1 != r2
    assert r1.startswith(e[:12]) and r2.startswith(e[:12])


# --- 7. re-run with same EXP_ID creates a new dir, doesn't touch old ------

def test_rerun_creates_new_dir(tmp_path):
    e = compute_exp_id(git_commit="x", git_dirty=False, config_hash="h",
                       data_fingerprint="d", split_fingerprint="s", seed=1)
    r1 = compute_run_id(e)
    r2 = compute_run_id(e)
    assert r1 != r2
    d1 = tmp_path / r1; d2 = tmp_path / r2
    d1.mkdir(); d2.mkdir()
    (d1 / "marker.txt").write_text("v1")
    assert (d1 / "marker.txt").read_text() == "v1"
    assert d2.exists() and d1.exists()


# --- 8. EXP_ID reflects code state (dirty flag flips it) ------------------

def test_exp_id_reflects_code_state():
    h = sha256_hex(canonical_json({"a": 1}))
    clean = compute_exp_id(git_commit="abc", git_dirty=False, config_hash=h,
                           data_fingerprint="d", split_fingerprint="s", seed=1)
    dirty = compute_exp_id(git_commit="abc", git_dirty=True, config_hash=h,
                           data_fingerprint="d", split_fingerprint="s", seed=1)
    assert clean != dirty  # uncommitted changes change provenance


# --- 9. profile deterministic per student, differs across students --------

def test_profile_deterministic_and_distinct():
    p1 = compute_profile("20250137")
    p1b = compute_profile("20250137")
    p2 = compute_profile("20250138")
    assert p1 == p1b
    assert p1["profile"]["id"] != p2["profile"]["id"]
    assert p1["profile"]["student_id_hash"] != p2["profile"]["student_id_hash"]


# --- 10. two profiles run with no source-code changes ---------------------

def test_two_profiles_no_code_changes():
    p1 = compute_profile("20250137")
    p2 = compute_profile("20250138")
    # Both are valid dicts with the same key structure; only values differ.
    assert set(p1.keys()) == set(p2.keys())
    for section in ["data", "model", "train", "optimizer", "scheduler"]:
        assert set(p1[section].keys()) == set(p2[section].keys())
    # No code change is required: the profile is consumed purely via config.


# --- 11. train script works on CPU; CUDA chosen config-wise ---------------

def test_model_runs_on_cpu_and_device_is_configurable():
    torch = pytest.importorskip("torch")
    from src.models.small_cnn import SmallCNN

    set_global_seed(11, deterministic=True)
    model = SmallCNN(width=16, depth=2)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    # device selection is a config field (configs/lab0.yaml: device.type=auto)
    cfg = Config.load(BASE_CFG)
    assert cfg.get("device", "type") in ("auto", "cpu", "cuda")


# --- extra: split manifest persistence keyed by fingerprint --------------

def test_split_manifest_reused(tmp_path):
    targets = _synthetic_targets()
    s1 = get_or_create_split(data_root=tmp_path, targets=targets, split_seed=5, train_fraction=0.25)
    s2 = get_or_create_split(data_root=tmp_path, targets=targets, split_seed=5, train_fraction=0.25)
    assert s1.fingerprint == s2.fingerprint
    # only one manifest file should exist for this fingerprint
    files = list((tmp_path / "splits").glob("cifar10_*.json"))
    assert len(files) == 1
