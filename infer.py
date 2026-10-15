"""Export one processed scenario's closed-loop rollout without metric dependencies."""
from argparse import ArgumentParser
from pathlib import Path
import json
import logging


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="One processed WOMD scenario .pkl")
    parser.add_argument("--output", required=True, help="Output .npz (existing files are refused)")
    parser.add_argument("--future_steps", type=int, default=300, help="Future frames at 10 Hz")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--disable_insertion", action="store_true")
    args = parser.parse_args(argv)
    if args.future_steps < 5 or args.future_steps % 5:
        parser.error("--future_steps must be a positive multiple of five frames")
    source, checkpoint, output = map(Path, (args.input, args.checkpoint, args.output))
    if not source.is_file() or not checkpoint.is_file():
        parser.error("Input scenario and checkpoint must both exist")
    if output.suffix != ".npz":
        parser.error("--output must end with .npz")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")

    import pickle
    import easydict
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from torch_geometric.data import Batch
    from infgen.datasets.scalable_dataset import WaymoTargetBuilder
    from infgen.model.infgen import InfGen

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu or use an NVIDIA CUDA host")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("infgen.inference")
    pl.seed_everything(args.seed, workers=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = easydict.EasyDict(saved["hyper_parameters"]["model_config"])
    model_config.num_recurrent_steps_val = args.future_steps
    model = InfGen(model_config, logger=logger)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.encoder.agent_encoder.disable_insertion = args.disable_insertion
    model.eval().to(args.device)
    with source.open("rb") as stream:
        scenario = pickle.load(stream)
    transformed = WaymoTargetBuilder(model.num_historical_steps, model.num_future_steps,
                                    max_num=32, training=False)(scenario)
    batch = Batch.from_data_list([transformed]).to(args.device)
    with torch.inference_mode():
        batch = model.token_processer(batch)
        batch = model.match_token_map(batch)
        batch = model.sample_pt_pred(batch)
        batch = model._fetch_enterings(batch)
        batch["batch_size_a"] = batch["agent"]["ptr"].diff()
        batch["batch_size_pl"] = batch["pt_token"]["ptr"].diff()
        batch["agent"]["av_index"] += batch["agent"]["ptr"][:-1]
        rollout = model.encoder.inference(batch)
    keys = ("agent_id", "pred_traj", "pred_head", "pred_type", "pred_state", "pred_z",
            "pred_shape", "pred_valid", "next_token_idx", "next_state_idx")
    arrays = {key: rollout[key].detach().cpu().numpy() for key in keys}
    if arrays["pred_traj"].shape[1] != args.future_steps + model.num_historical_steps:
        raise RuntimeError("Decoder returned an unexpected rollout length")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Decoder returned non-finite output")
    arrays["scenario_id"] = np.asarray(str(scenario["scenario_id"]))
    arrays["ego_index"] = np.asarray(rollout["ego_index"], dtype=np.int64)
    arrays["seed"] = np.asarray(args.seed, dtype=np.int64)
    arrays["paper_objective"] = np.asarray(model_config.get("paper_objective", False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps({"output": str(output.resolve()), "agents": arrays["pred_traj"].shape[0],
                      "frames": arrays["pred_traj"].shape[1], "seed": args.seed}))


if __name__ == "__main__":
    main()
