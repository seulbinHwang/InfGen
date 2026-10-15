"""Guard the scientific settings and explicitly documented release differences."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_profile():
    return yaml.safe_load((ROOT / "configs/paper_reproduction.yaml").read_text())


def test_shared_settings_reach_dataset_model_and_decoder():
    config = load_profile()
    for section in (config["Dataset"], config["Model"], config["Model"]["decoder"]):
        assert section["num_historical_steps"] == 11
        assert section["num_future_steps"] == 80
        assert section["token_size"] == 2048
        assert section["predict_motion"] is True
        assert section["predict_state"] is True
        assert section["predict_map"] is False
        assert section["predict_occ"] is True
        assert section["state_token"] == {"invalid": 0, "valid": 1, "enter": 2, "exit": 3}
    assert config["Dataset"]["train_batch_size"] * config["Trainer"]["devices"] == 8


def test_paper_loss_coefficients_and_control_class_order():
    model = load_profile()["Model"]
    assert model["paper_objective"] is True
    weights = model["loss_weight"]
    for name, expected in {
        "token_cls_loss": 1, "pos_cls_loss": 10, "head_cls_loss": 1,
        "state_cls_loss": 10, "shape_reg_loss": 0.2, "type_cls_loss": 5,
    }.items():
        assert weights[name] == expected
    assert weights["state_weight"] == [0.0, 0.1, 0.9]  # invalid / KEEP / REMOVE
    assert weights["seed_state_weight"] == [0.9, 0.1]  # BEGIN MOTION / ADD
    assert weights["seed_type_weight"] == [1.0, 1.0, 1.0]
    for name in ("map_token_loss", "offset_reg_loss", "pos_reg_loss", "head_reg_loss",
                 "agent_occ_loss", "pt_occ_loss"):
        assert weights[name] == 0


def test_table_four_dimensions_and_raw_frame_temporal_window():
    model = load_profile()["Model"]
    decoder = model["decoder"]
    assert model["hidden_dim"] == model["num_heads"] * model["head_dim"] == 128
    assert model["num_freq_bands"] == 64
    assert decoder["num_agent_layers"] == 6
    assert decoder["num_map_layers"] == 3
    assert decoder["time_span"] // 5 == 12
    assert (decoder["a2a_radius"], decoder["pl2a_radius"], decoder["pl2pl_radius"]) == (60, 30, 10)
    assert (decoder["a2sa_radius"], decoder["pl2sa_radius"], decoder["pl2seed_radius"]) == (10, 10, 75)
    assert model["num_recurrent_steps_val"] == 300
    assert model["val_close_loop"] is True
    assert model["val_insert"] is False  # Insertion-only diagnostics bypass full rollout.


def test_actual_grid_difference_is_documented_instead_of_silently_changed():
    from infgen.modules.attr_tokenizer import Attr_Tokenizer

    model = load_profile()["Model"]
    tokenizer = Attr_Tokenizer(
        model["grid_range"], model["grid_interval"],
        model["pl2seed_radius"], model["angle_interval"],
    )
    assert tokenizer.grid_size == 1961
    assert tokenizer.angle_size == 120
    audit = (ROOT / "docs/paper_audit.md").read_text()
    assert "1849" in audit and "1961" in audit
