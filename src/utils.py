import torch
import numpy as np 
import random 

from typing import Union, Tuple, Dict
import os
import json

import omegaconf
from omegaconf import OmegaConf

def get_class_by_name(module, model_name : str):
    model = None
    if hasattr(module, model_name):
        model = getattr(module, model_name)
    return model

def get_config(
        exp_path: str,
        cfg_path: str = './configs/'
        ):
    exp_path_dir = cfg_path + 'experiments/'
    path = cfg_path + exp_path_dir
    files = [cfg_path+fl for fl in os.listdir(cfg_path) if 'yaml' in fl]
    base_cfg = [omegaconf.OmegaConf.load(cfg) for cfg in files]
    base_cfg = omegaconf.OmegaConf.merge(*base_cfg)

    exp_cfg = omegaconf.OmegaConf.load(exp_path_dir + exp_path)
    cfg = omegaconf.OmegaConf.merge(base_cfg, exp_cfg)
    return cfg

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_tokenizer(config: omegaconf.DictConfig):
    module = __import__(config.tokenizer.module_name)
    tokenizer_class = get_class_by_name(module, config.tokenizer.class_name)
    return tokenizer_class.from_pretrained(config.model_name)

def read_dataset(data_path: str, debug=False) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    test_file = 'test.json' if debug is False else 'data_sample.json'
    train_file = 'train.json'
    files = [train_file, test_file]
    dataset = []
    for fl in files:
        with open(os.path.join(data_path, fl), 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset.append(data)
    return tuple(dataset) # (train, test)