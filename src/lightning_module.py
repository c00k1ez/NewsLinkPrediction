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
from src.metrics import F1_score, Recall_at_k


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

        self.conf_matrix = ConfusionMatrix(num_classes=2)
        self.f1_score = F1_score()
        self.recall_at_1 = Recall_at_k(k=1)
        self.recall_at_3 = Recall_at_k(k=3)
        self.recall_at_5 = Recall_at_k(k=5)
    
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
        assert loss.detach().cpu().isnan() != True
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch)
        loss = self.criterion(outputs)
        labels = batch['label']
        ret = {'loss_val': loss}
        # shape : [batch_size, ]
        sim = torch.nn.functional.cosine_similarity(outputs['anchor'], outputs['positive'])
        sim[sim >= self.hparams.cos_margin] = 1
        sim[sim < self.hparams.cos_margin] = 0
        sim = sim.type_as(labels)
        # compute confusion matrix
        matr = self.conf_matrix(sim, labels)
        assert list(matr.shape) == [2, 2]
        # compute recalls per batch
        self.recall_at_1(outputs)
        self.recall_at_3(outputs)
        self.recall_at_5(outputs)
        ret['confusion_matrix'] = matr
        return ret

    def validation_epoch_end(self, outputs):
        matr = sum([output['confusion_matrix'] for output in outputs])
        loss_val = torch.stack([x['loss_val'] for x in outputs]).mean()
        # compute F1 score
        total_f1, [f1_0, f1_1] = self.f1_score(matr)
        # compute recall@k scores
        recall_at_1 = self.recall_at_1.compute_metric()
        recall_at_3 = self.recall_at_3.compute_metric()
        recall_at_5 = self.recall_at_5.compute_metric()
        logging.info('log confusion matrix at {} step: \n {}'.format(self.global_step, np.matrix(matr.tolist())))
        #print('log confusion matrix at {} step: {} \n'.format(self.global_step, np.matrix(matr.tolist())))
        output = {
            'val_loss': loss_val,
            'log': {
                'val_loss': loss_val,
                'weighted_f1': total_f1,
                'f1_0': f1_0,
                'f1_1': f1_1,
                'recall@1': recall_at_1,
                'recall@3': recall_at_3,
                'recall@5': recall_at_5
            }
        }
        return output
