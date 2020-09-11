import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import ConfusionMatrix
import transformers
from transformers import get_linear_schedule_with_warmup
import numpy as np

import src.models as models
import src.losses as losses
from src.utils import get_class_by_name

class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningModel, self).__init__()
        logging.basicConfig(filename='logs/{}.log'.format(hparams.experiment_name), level=logging.INFO)

        self.hparams = hparams
        backbone_model_class = get_class_by_name(transformers, hparams.backbone_model)
        backbone_model = backbone_model_class.from_pretrained(hparams.model_name)
        if hasattr(hparams, 'freeze_backbone') and hparams.freeze_backbone is True:
            for param in backbone_model.parameters():
                param.requires_grad = False
        self.siamese_model = get_class_by_name(models, hparams.main_model)(backbone_model, **hparams['model'])

        criterion_class = get_class_by_name(losses, hparams.criterion_name)
        if 'criterion' in hparams:
            self.criterion = criterion_class(**hparams['criterion'])
        else:
            self.criterion = criterion_class()

        self.conf_matrix = ConfusionMatrix()
    
    def configure_optimizers(self):
        opt = transformers.AdamW(self.siamese_model.parameters(), **self.hparams['optimizer'])
        output = opt
        if 'scheduler' in self.hparams:
            scheduler = get_linear_schedule_with_warmup(opt, **self.hparams['scheduler'])
            output = ([opt], [scheduler])
        return output
    

    def forward(self, batch):
        return self.siamese_model(batch)

    def training_step(self, batch, batch_nb):
        outputs = self.forward(batch)
        loss = self.criterion(outputs)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch)
        loss = self.criterion(outputs)
        labels = batch['label']
        ret = {'loss_val': loss}
        if self.hparams.model_output_prob is False:
            # shape : [batch_size, ]
            sim = torch.nn.functional.cosine_similarity(outputs['anchor'], outputs['positive'])
            sim[sim >= self.hparams.cos_margin] = 1
            sim[sim < self.hparams.cos_margin] = 0
            sim = sim.type_as(labels)
            matr = self.conf_matrix(sim, labels)
            assert list(matr.shape) == [2, 2]
            ret['confusion_matrix'] = matr
        return ret

    def f1_score(self, matr: torch.Tensor, average='weighted'):
        assert average in ['weighted',]
        tn, fp, fn, tp = matr[0, 0], matr[0, 1], matr[1, 0], matr[1, 1]
        precision_0, recall_0 = tn / (tn + fn), tn / (tn + fp)
        precision_1, recall_1 = tp / (tp + fp), tp / (tp + fn)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
        weight_0 = (tn + fp) / (tn + fp + fn + tp)
        weight_1 = (tp + fn) / (tn + fp + fn + tp)
        total_f1 = weight_0 * f1_0 + weight_1 * f1_1
        return total_f1, [f1_0, f1_1]

    def validation_epoch_end(self, outputs):
        matr = sum([output['confusion_matrix'] for output in outputs])
        loss_val = torch.stack([x['loss_val'] for x in outputs]).mean()
        total_f1, [f1_0, f1_1] = self.f1_score(matr)
        logging.info('log confusion matrix at {} step: \n {}'.format(self.global_step, np.matrix(matr.tolist())))
        #print('log confusion matrix at {} step: {} \n'.format(self.global_step, np.matrix(matr.tolist())))
        output = {
            'val_loss': loss_val,
            'log': {
                'val_loss': loss_val,
                'weighted_f1': total_f1,
                'f1_0': f1_0,
                'f1_1': f1_1
            }
        }
        return output
