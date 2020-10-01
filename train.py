import comet_ml

import argparse

import torch 
import src.models as models
from src.utils import get_class_by_name, get_config, seed_all, init_tokenizer, read_dataset
from src.dataset import PairsDataset
from src.lightning_module import LightningModel
from src.callbacks import UnfreezingCallback

import pytorch_lightning as pl
import transformers

import os

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_config', type=str, default='baseline_bce_model.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_comet', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--distributed_backend', type=str, default=None, choices=[None, 'ddp', 'ddp_cpu', 'dp'])
    parser.add_argument('--fast_dev_run', type=bool, default=False)

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    # -----------------------------------------------------
    # step 1 : init config
    print('Use {} config'.format(args.experiment_config))
    config = get_config(args.experiment_config)
    # -----------------------------------------------------
    # step 2 : init tokenizer
    tokenizer = init_tokenizer(config)
    # -----------------------------------------------------
    # step 3 : init loaders
    train, test = read_dataset(config.data_path, debug=False)
    train = PairsDataset(train, tokenizer, **config['datasets'])
    test = PairsDataset(test, tokenizer, mode='test', **config['datasets'])
    loaders = {
        'train_dataloader' : torch.utils.data.DataLoader(train, **config['loaders']),
        'val_dataloaders' : torch.utils.data.DataLoader(test, **config['loaders'])
    }
    if 'scheduler' in config:
        config['scheduler'].num_training_steps = len(loaders['train_dataloader']) * config['trainer'].max_epochs
    # -----------------------------------------------------
    # step 4 : init model
    model = LightningModel(config)
    # -----------------------------------------------------
    # step 5 : init logger(s)
    logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=1,
            name='lightning_logs'
        )
    if 'logger' in config and args.use_comet:
        comet_logger = pl.loggers.CometLogger(**config['logger'], experiment_name=config.experiment_name)
        logger = [logger, comet_logger]
    
    # -----------------------------------------------------
    # step 6 : init callbacks
    callbacks = None
    if 'callbacks' in config:
        callbacks = [UnfreezingCallback(**config['callbacks']),]
    # -----------------------------------------------------
    # step 7 : init Trainer
    gpus = '0'
    if args.gpu_id is not None:
        gpus = [args.gpu_id,]
    trainer = pl.Trainer(
        **config['trainer'],
        logger=logger,
        gpus=gpus,
        distributed_backend=args.distributed_backend,
        fast_dev_run=args.fast_dev_run,
        callbacks=callbacks
    )
    # -----------------------------------------------------
    # step 8 : train model
    trainer.fit(model, **loaders)
