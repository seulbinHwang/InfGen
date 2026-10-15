"""Train or evaluate the official InfGen model with explicit, portable options."""
from argparse import ArgumentParser
from pathlib import Path
import json
import logging

REPO = Path(__file__).resolve().parent


def build_parser(default_action=None):
    parser = ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=default_action is None)
    for action in ("train", "validate", "test", "plot_rollouts"):
        actions.add_argument(f"--{action}", action="store_true")
    parser.set_defaults(default_action=default_action)
    parser.add_argument("--config", default=str(REPO / "configs/paper_reproduction.yaml"))
    parser.add_argument("--data_root", help="Processed data root containing training/ and validation/.")
    parser.add_argument("--ckpt_path", help="Resume all training states, or load a model for evaluation.")
    parser.add_argument("--pretrain_ckpt", help="Explicit partial parameter initialization (not a resume).")
    parser.add_argument("--save_ckpt_path", default="output/infgen")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "auto"))
    parser.add_argument("--devices", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--batch_size", type=int, help="Per-device training batch size.")
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_train_batches", type=int)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--scenario_id")
    parser.add_argument("--wandb", action="store_true", help="Explicitly enable optional remote logging.")
    return parser


def main(argv=None, default_action=None):
    parser = build_parser(default_action)
    args = parser.parse_args(argv)
    action = next((name for name in ("train", "validate", "test", "plot_rollouts")
                   if getattr(args, name)), args.default_action)
    if args.ckpt_path and not Path(args.ckpt_path).is_file():
        parser.error(f"Checkpoint does not exist: {args.ckpt_path}")
    if action != "train" and not args.ckpt_path:
        parser.error("Evaluation requires --ckpt_path; use infer.py for rollout export.")
    if args.pretrain_ckpt and args.ckpt_path:
        parser.error("Choose partial initialization or full resume, not both.")

    import pytorch_lightning as pl
    import torch
    import yaml
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
    from pytorch_lightning.strategies import DDPStrategy
    from infgen.datasets.scalable_dataset import MultiDataModule
    from infgen.model.infgen import InfGen
    from infgen.utils.func import load_config_act

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("infgen")
    pl.seed_everything(args.seed, workers=True)
    config = load_config_act(args.config)
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
        config.Dataset.train_raw_dir = str(data_root / "training")
        config.Dataset.val_raw_dir = str(data_root / "validation")
        config.Dataset.test_raw_dir = str(data_root / "validation")
        config.Dataset.val_tfrecords_splitted = None
    for key in ("train_raw_dir", "val_raw_dir", "test_raw_dir", "val_tfrecords_splitted"):
        value = config.Dataset.get(key)
        if value:
            values = value if isinstance(value, list) else [value]
            resolved = [str((REPO / Path(v).expanduser()).resolve()) for v in values]
            config.Dataset[key] = resolved if isinstance(value, list) else resolved[0]
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num_workers must be nonnegative")
        config.Dataset.num_workers = args.num_workers
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch_size must be positive")
        config.Dataset.train_batch_size = args.batch_size
    for key in ("accelerator", "devices", "max_epochs"):
        value = getattr(args, key)
        if value is not None:
            config.Trainer[key] = value
    if config.Trainer.devices < 1 or config.Trainer.max_epochs < 1:
        parser.error("devices and max_epochs must be positive")
    if config.Trainer.accelerator == "gpu" and not torch.cuda.is_available():
        parser.error("The GPU recipe requires NVIDIA CUDA; local checks use --accelerator cpu --devices 1")
    if args.max_epochs is not None:
        config.Model.total_steps = args.max_epochs
    if action != "train" and config.Dataset.val_batch_size != 1:
        parser.error("Released closed-loop inference supports one scenario per batch.")
    if config.Trainer.accelerator == "cpu":
        config.Dataset.pin_memory = False
    output = Path(args.save_ckpt_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w") as stream:
        yaml.safe_dump(json.loads(json.dumps(config)), stream, sort_keys=False)
    run_logger = WandbLogger(project="infgen") if args.wandb else CSVLogger(str(output), name="logs")
    strategy = (DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
                if config.Trainer.devices * config.Trainer.num_nodes > 1 else "auto")
    model = InfGen(config.Model, save_path=str(output), logger=logger,
                   max_epochs=config.Trainer.max_epochs)
    if args.pretrain_ckpt:
        model.load_state_from_file(args.pretrain_ckpt, to_cpu=True)
    datamodule = MultiDataModule(**config.Dataset, logger=logger, scenario_id=args.scenario_id)
    checkpoint = ModelCheckpoint(dirpath=str(output), filename="epoch-{epoch:02d}",
                                 save_last=True, save_top_k=-1, save_on_train_epoch_end=True)
    trainer_kwargs = dict(
        accelerator=config.Trainer.accelerator, devices=config.Trainer.devices,
        strategy=strategy, num_nodes=config.Trainer.num_nodes,
        max_epochs=config.Trainer.max_epochs, max_steps=args.max_steps,
        accumulate_grad_batches=config.Trainer.accumulate_grad_batches,
        precision=config.Trainer.precision,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        logger=run_logger, num_sanity_val_steps=0, log_every_n_steps=1,
        gradient_clip_val=0.5,
    )
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    trainer = pl.Trainer(**trainer_kwargs)
    if action == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    else:
        model.save_path = str(Path(args.save_path or output / "rollouts").resolve())
        Path(model.save_path).mkdir(parents=True, exist_ok=True)
        model.set({"validate": "validation", "test": "test", "plot_rollouts": "plot_rollouts"}[action])
        trainer.validate(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
