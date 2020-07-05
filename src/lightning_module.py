import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import ConfusionMatrix
import transformers

import src.models as models
import src.losses as losses
from src.utils import get_class_by_name

class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningModel, self).__init__()
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
        return transformers.AdamW(self.siamese_model.parameters(), **self.hparams['optimizer'])

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
        labels = batch['label'].cpu()
        ret = {'loss_val': loss}
        if self.hparams.model_output_prob is False:
            # shape : [batch_size, ]
            sim = torch.nn.functional.cosine_similarity(outputs['anchor'], outputs['positive'])
            sim[sim >= self.hparams.cos_margin] = 1
            sim[sim < self.hparams.cos_margin] = 0
            sim = sim.cpu()
            sim = sim.type(torch.LongTensor)
            matr = self.conf_matrix(sim, labels)
            ret['confusion_matrix'] = matr
        return ret

    def f1_score(self, matr: torch.Tensor):
        pr_1 = matr[0, 0] / (matr[0, 0] + matr[0, 1])
        rec_1 = matr[0, 0] / (matr[0, 0] + matr[1, 0])
        pr_2 = matr[1, 1] / (matr[1, 1] + matr[1, 0])
        rec_2 = matr[1, 1] / (matr[1, 1] + matr[0, 1])
        f1_1 = 2 * pr_1 * rec_1 / (pr_1 + rec_1)
        f1_2 = 2 * pr_2 * rec_2 / (pr_2 + rec_2)
        return (f1_1 + f1_2) / 2

    def validation_epoch_end(self, outputs):
        matr = sum([output['confusion_matrix'] for output in outputs])
        loss_val = torch.stack([x['loss_val'] for x in outputs]).mean()
        f1 = self.f1_score(matr)
        output = {
            'val_loss': loss_val,
            'macro_f1': f1,
            'log': {
                'val_loss': loss_val, 
                'macro_f1': f1
            }
        }

        return output
