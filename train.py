import argparse
import logging
import os

import comet_ml
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.callbacks import UnfreezingCallback
from src.dataset import PairsDataset
from src.lightning_module import LightningModel
from src.utils import get_config, init_tokenizer, read_dataset


logging.getLogger("transformers").setLevel(logging.CRITICAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, default="softmax_loss_baseline_bert.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_comet", dest="use_comet", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--distributed_backend", type=str, default=None, choices=[None, "ddp", "ddp_cpu", "dp"])
    parser.add_argument("--fast_dev_run", dest="fast_dev_run", action="store_true")
    parser.add_argument("--resume_from_checkpoint", dest="resume_from_checkpoint", action="store_true")
    parser.set_defaults(fast_dev_run=False, use_comet=False, resume_from_checkpoint=False)

    args = parser.parse_args()

    assert args.gpu_id is None or args.gpus is None

    pl.seed_everything(args.seed)
    # -----------------------------------------------------
    # step 1 : init config
    print("Use {} config".format(args.experiment_config))
    config = get_config(args.experiment_config)
    # -----------------------------------------------------
    # step 2 : init tokenizer
    tokenizer = init_tokenizer(config)
    # -----------------------------------------------------
    # step 3 : init loaders
    train, test, val = read_dataset(config.data_path, debug=False)
    train = PairsDataset(train, tokenizer, **config["datasets"])
    val = PairsDataset(val, tokenizer, mode="test", **config["datasets"])
    test = PairsDataset(test, tokenizer, mode="test", **config["datasets"])
    loaders = {
        "train_dataloader": torch.utils.data.DataLoader(train, **config["loaders"], drop_last=True),
        "val_dataloaders": torch.utils.data.DataLoader(val, **config["loaders"]),
    }
    test_loader = torch.utils.data.DataLoader(test, **config["loaders"])
    if "scheduler" in config:
        config["scheduler"].num_training_steps = len(loaders["train_dataloader"]) * config["trainer"].max_epochs
    # -----------------------------------------------------
    # step 4 : init model
    model = LightningModel(config)
    # -----------------------------------------------------
    # step 5 : init logger(s)
    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    if "logger" in config and args.use_comet:
        comet_logger = pl.loggers.CometLogger(**config["logger"], experiment_name=config.experiment_name)
        comet_logger.experiment.add_tag("Recall")
        logger = [logger, comet_logger]

    # -----------------------------------------------------
    # step 6 : init callbacks
    callbacks = None
    if "callbacks" in config:
        callbacks = [
            UnfreezingCallback(**config["callbacks"]),
        ]

    dirname = f"{config.ckeckpoints_dir}/{config.experiment_name}_seed_{str(args.seed)}/"
    filename = config.experiment_name + "_seed_" + str(args.seed) + "_{epoch}-{recall_at_1:.3f}"
    checkpoint_callback = ModelCheckpoint(
        dirname=dirname, filename=filename, save_top_k=1, verbose=True, monitor="recall_at_1", mode="max"
    )
    if callbacks is None:
        callbacks = [
            checkpoint_callback,
        ]
    else:
        callbacks = callbacks + [
            checkpoint_callback,
        ]
    # -----------------------------------------------------
    # step 7 : init Trainer
    gpus = None
    if args.gpu_id is not None:
        gpus = [
            args.gpu_id,
        ]
    if args.gpus is not None:
        gpus = args.gpus

    if config.trainer.default_root_dir is not None:
        config.trainer.default_root_dir = config.trainer.default_root_dir + "/" + config.experiment_name

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        resume_from_checkpoint = dirname + "/" + os.listdir(dirname)[0]
    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        gpus=gpus,
        distributed_backend=args.distributed_backend,
        fast_dev_run=args.fast_dev_run,
        callbacks=callbacks,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    # -----------------------------------------------------
    # step 8 : train model
    trainer.fit(model, **loaders)
    # -----------------------------------------------------
    # step 9 : validate model using test set
    checkpoint_path = dirname + "/" + os.listdir(dirname)[0]
    model = LightningModel.load_from_checkpoint(checkpoint_path)
    trainer.test(model, test_loader)
