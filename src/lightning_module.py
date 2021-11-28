import logging

import pytorch_lightning as pl
import torch
import transformers
from torchmetrics import RetrievalRecall
from transformers import get_linear_schedule_with_warmup

import src.losses as losses
import src.models as models
from src.utils import get_class_by_name


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningModel, self).__init__()
        logging.basicConfig(filename="logs/{}.log".format(hparams.experiment_name), level=logging.INFO)

        self.save_hyperparameters(hparams)
        backbone_model_class = get_class_by_name(transformers, hparams.backbone_model)
        backbone_model = backbone_model_class.from_pretrained(hparams.model_name)
        if hasattr(hparams, "freeze_backbone") and hparams.freeze_backbone:
            for param in backbone_model.parameters():
                param.requires_grad = False
        self.siamese_model = get_class_by_name(models, hparams.main_model)(backbone_model, **hparams["model"])

        criterion_class = get_class_by_name(losses, hparams.criterion_name)
        if "criterion" in hparams:
            self.criterion = criterion_class(**hparams["criterion"])
        else:
            self.criterion = criterion_class()

        self.recall_at_1 = RetrievalRecall(k=1).clone()
        self.recall_at_3 = RetrievalRecall(k=3).clone()
        self.recall_at_5 = RetrievalRecall(k=5).clone()

    def configure_optimizers(self):
        opt = transformers.AdamW(self.siamese_model.parameters(), **self.hparams["optimizer"])
        output = opt
        if "scheduler" in self.hparams:
            scheduler = get_linear_schedule_with_warmup(opt, **self.hparams["scheduler"])
            output = ([opt], [scheduler])
        return output

    def forward(self, batch):
        return self.siamese_model(batch)

    def training_step(self, batch, batch_nb):
        outputs = self(batch)
        loss = self.criterion(outputs)
        # assert loss.detach().cpu().isnan() is not True
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_nb):
        outputs = self(batch)
        loss = self.criterion(outputs)
        # compute recalls per batch
        distance = outputs["anchor"] @ outputs["positive"].t()
        target = torch.eye(distance.shape[0]).type_as(distance).long()
        inds = torch.arange(target.shape[0]).unsqueeze(-1).repeat(1, distance.shape[0]).type_as(target)
        self.log(
            "recall_at_1",
            self.recall_at_1(distance, target, indexes=inds),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "recall_at_3",
            self.recall_at_3(distance, target, indexes=inds),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "recall_at_5",
            self.recall_at_5(distance, target, indexes=inds),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_nb):
        outputs = self(batch)
        loss = self.criterion(outputs)

        distance = outputs["anchor"] @ outputs["positive"].t()
        target = torch.eye(distance.shape[0]).type_as(distance).long()
        inds = torch.arange(target.shape[0]).unsqueeze(-1).repeat(1, distance.shape[0]).type_as(target)
        self.log(
            "test_recall_at_1",
            self.recall_at_1(distance, target, indexes=inds),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_recall_at_3",
            self.recall_at_3(distance, target, indexes=inds),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_recall_at_5",
            self.recall_at_5(distance, target, indexes=inds),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log("test_loss", loss, on_step=True, on_epoch=True)
