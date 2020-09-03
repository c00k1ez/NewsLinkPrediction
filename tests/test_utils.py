import torch

import src.utils as utils 
import src.models as models

import pytest

import omegaconf

import transformers

class TestUtils:
    def test_get_model(self):
        model_class = utils.get_class_by_name(models, 'BaselineSiameseNetwork')
        assert model_class in [None, models.BaselineSiameseNetwork]

    def test_get_config(self):
        cfg = utils.get_config('./configs/experiments/baseline_bce_model.yaml')
        assert type(cfg) == omegaconf.dictconfig.DictConfig

    def test_init_tokenizer(self):
        cfg = omegaconf.OmegaConf.create({
            'model_name': 'DeepPavlov/rubert-base-cased',
            'tokenizer': {
                'module_name': 'transformers',
                'class_name': 'AutoTokenizer'
            }
        })
        tokenizer = utils.init_tokenizer(cfg)
        assert type(tokenizer) == transformers.tokenization_bert.BertTokenizer

    def test_read_dataset(self):
        train, test = utils.read_dataset('./data/', debug=True)
        assert (type(train) == dict) and (type(test) == dict)