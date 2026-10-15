"""Run the full CPU train/save/resume/load/rollout contract on synthetic fixtures."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import torch

from make_synthetic_data import make_scenario

REPO = Path(__file__).resolve().parents[1]


def verify(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError(f"Verification directory must be empty: {folder}")
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    records = []

    def run(name, *arguments):
        command = [sys.executable, *map(str, arguments)]
        log_path = folder / f"{name}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError(f"{name} failed (exit {result.returncode}):\n{log_path.read_text()[-8000:]}")
        records.append({"name": name, "exit_code": 0})

    for split in ("training", "validation"):
        destination = folder / "data" / split
        destination.mkdir(parents=True)
        with (destination / "synthetic_0000.pkl").open("wb") as stream:
            pickle.dump(make_scenario(), stream)
    checkpoint_dir = folder / "checkpoints"
    common = [REPO / "train.py", "--data_root", folder / "data", "--accelerator", "cpu",
              "--devices", "1", "--num_workers", "0", "--limit_train_batches", "1",
              "--limit_val_batches", "0", "--save_ckpt_path", checkpoint_dir]
    run("train", *common, "--max_steps", "2")
    checkpoint = checkpoint_dir / "last.ckpt"
    initial = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert initial["global_step"] == 2
    probe_name = "encoder.agent_encoder.token_predict_head.mlp.0.weight"
    if probe_name not in initial["state_dict"]:
        probe_name = next(name for name, value in initial["state_dict"].items()
                          if value.is_floating_point() and value.ndim == 2 and "token_predict_head" in name)
    before = initial["state_dict"][probe_name].clone()
    del initial
    run("resume", *common, "--max_steps", "3", "--ckpt_path", checkpoint)
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert resumed["global_step"] == 3
    assert not torch.equal(before, resumed["state_dict"][probe_name]), "Resume did not update weights"
    assert all(torch.isfinite(value).all() for value in resumed["state_dict"].values()
               if value.is_floating_point()), "Non-finite model parameters"
    for optimizer in resumed["optimizer_states"]:
        assert all(torch.isfinite(value).all() for state in optimizer["state"].values()
                   for value in state.values() if torch.is_tensor(value)), "Non-finite optimizer state"
    del resumed
    numeric_logs = 0
    for path in checkpoint_dir.glob("logs/version_*/metrics.csv"):
        for row in csv.DictReader(path.open()):
            for value in row.values():
                if value:
                    assert np.isfinite(float(value)), f"Non-finite metric in {path}"
                    numeric_logs += 1
    assert numeric_logs > 0
    source = folder / "data/validation/synthetic_0000.pkl"
    inference = [REPO / "infer.py", "--checkpoint", checkpoint, "--input", source,
                 "--device", "cpu", "--seed", "2024"]
    run("rollout_80", *inference, "--future_steps", "80", "--output", folder / "rollout_80.npz")
    run("rollout_300", *inference, "--future_steps", "300", "--output", folder / "rollout_300.npz")
    run("repeat_80", *inference, "--future_steps", "80", "--output", folder / "repeat_80.npz")
    changed = copy.deepcopy(make_scenario())
    torch.manual_seed(99)
    for key in ("position", "heading", "velocity", "shape"):
        changed["agent"][key][:, 11:] = torch.randn_like(changed["agent"][key][:, 11:]) * 100
    changed["agent"]["valid_mask"][:, 11:] = False
    changed["agent"]["valid_mask"][6, 11:] = True  # Future-only agent must not become an observed seed.
    variant = folder / "future_changed.pkl"
    with variant.open("wb") as stream:
        pickle.dump(changed, stream)
    run("future_invariance", REPO / "infer.py", "--checkpoint", checkpoint, "--input", variant,
        "--output", folder / "future_changed.npz", "--future_steps", "80", "--seed", "2024")
    with np.load(folder / "rollout_80.npz", allow_pickle=False) as base:
        assert base["pred_traj"].shape[1] == 91
        for other in ("repeat_80.npz", "future_changed.npz"):
            with np.load(folder / other, allow_pickle=False) as comparison:
                for key in base.files:
                    assert np.array_equal(base[key], comparison[key]), f"Output changed: {other}/{key}"
    with np.load(folder / "rollout_300.npz", allow_pickle=False) as long:
        assert long["pred_traj"].shape[1] == 311
    report = {"status": "passed", "checks": records, "numeric_metrics_checked": numeric_logs,
              "limitations": "Synthetic CPU contract only; no WOMD scores or CUDA/DDP verification."}
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, help="Optional empty directory to keep logs/checkpoints")
    args = parser.parse_args()
    if args.workdir:
        verify(args.workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="infgen-verification-") as directory:
            verify(directory)


if __name__ == "__main__":
    main()
