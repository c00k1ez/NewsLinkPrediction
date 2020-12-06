import comet_ml

import argparse

import omegaconf

import torch 
import src.models as models
from src.utils import get_class_by_name, get_config, seed_all, init_tokenizer, read_dataset
from src.dataset import PairsDataset
from src.lightning_module import LightningModel
from src.callbacks import UnfreezingCallback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import transformers

import os

import logging

logging.getLogger("transformers").setLevel(logging.CRITICAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_name', type=str, default='softmax_loss_baseline_bert_seed_42')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_comet', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--distributed_backend', type=str, default=None, choices=[None, 'ddp', 'ddp_cpu', 'dp'])
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--experiment_key', type=str, default=None)

    args = parser.parse_args()

    assert args.experiment_key is not None

    assert args.gpu_id is None or args.gpus is None

    pl.seed_everything(args.seed)

    full_ckpt_path = './checkpoints/' + args.checkpoint_name + '/'
    full_ckpt_path = full_ckpt_path + os.listdir(full_ckpt_path)[0]

    config = torch.load(full_ckpt_path)['hyper_parameters']

    tokenizer = init_tokenizer(config)

    train, test, val = read_dataset(config.data_path, debug=False)
    train = PairsDataset(train, tokenizer, **config['datasets'])
    val = PairsDataset(val, tokenizer, mode='test', **config['datasets'])
    test = PairsDataset(test, tokenizer, mode='test', **config['datasets'])
    loaders = {
        'train_dataloader' : torch.utils.data.DataLoader(train, **config['loaders'], drop_last=True),
        'val_dataloaders' : torch.utils.data.DataLoader(val, **config['loaders'])
    }
    test_loader = torch.utils.data.DataLoader(test, **config['loaders'])
    if 'scheduler' in config:
        config['scheduler'].num_training_steps = len(loaders['train_dataloader']) * config['trainer'].max_epochs
    
    model = LightningModel.load_from_checkpoint(full_ckpt_path)

    logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=1,
            name='lightning_logs'
        )
    if 'logger' in config and args.use_comet:
        comet_logger = pl.loggers.CometLogger(**config['logger'], experiment_name=config.experiment_name)
        comet_logger.experiment = comet_ml.ExistingExperiment(**config['logger'], previous_experiment=args.experiment_key)
        #comet_logger.experiment.add_tag('Recall')
        logger = [logger, comet_logger]
    
    gpus = None
    if args.gpu_id is not None:
        gpus = [args.gpu_id,]
    if args.gpus is not None:
        gpus = args.gpus

    trainer = pl.Trainer(
        **config['trainer'],
        logger=logger,
        gpus=gpus,
        distributed_backend=args.distributed_backend,
        fast_dev_run=args.fast_dev_run,
        resume_from_checkpoint=full_ckpt_path
    )

    trainer.fit(model, **loaders)

    model = LightningModel.load_from_checkpoint(full_ckpt_path)
    trainer.test(model, test_loader, full_ckpt_path)