import argparse

import torch 
import src.models as models
from src.utils import get_class_by_name, get_config, seed_all, init_tokenizer, read_dataset
from src.dataset import PairsDataset
from src.lightning_module import LightningModel

import pytorch_lightning as pl
import transformers

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_config', type=str, default='./configs/experiments/baseline_bce_model.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='./data/')
    args = parser.parse_args()
    # -----------------------------------------------------
    # step 1 : init config
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
    # -----------------------------------------------------
    # step 4 : init model
    model = LightningModel(config)
    # -----------------------------------------------------
    # step 5 : init logger(s)
    logger = False
    if 'logger' in config:
        logger = pl.loggers.CometLogger(**config['logger'], experiment_name=config.experiment_name)
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=1,
            name='lightning_logs'
        )
    # -----------------------------------------------------
    # step 6 : init logger(s)
    trainer = pl.Trainer(**config['trainer'], logger=logger)
    # -----------------------------------------------------
    # step 7 : train model
    trainer.fit(model, **loaders)
