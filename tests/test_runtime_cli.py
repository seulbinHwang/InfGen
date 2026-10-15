import subprocess
import sys
from pathlib import Path

from run import build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_repository_relative():
    args = build_parser("train").parse_args([])
    assert Path(args.config) == ROOT / "configs/paper_reproduction.yaml"
    assert args.devices is None  # YAML default is not silently overridden.


def test_explicit_actions_are_mutually_exclusive():
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--train", "--test"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "not allowed" in result.stderr


def test_inference_refuses_missing_checkpoint_before_importing_model(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "infer.py"), "--checkpoint", "absent.ckpt",
                             "--input", "absent.pkl", "--output", str(tmp_path / "out.npz")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "must both exist" in result.stderr


def test_inference_rejects_partial_token_horizon(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "infer.py"), "--checkpoint", "absent.ckpt",
                             "--input", "absent.pkl", "--output", str(tmp_path / "out.npz"),
                             "--future_steps", "3"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "multiple of five" in result.stderr
