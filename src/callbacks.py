import pytorch_lightning as pl
import torch


class UnfreezingCallback(pl.Callback):
    def __init__(self, unfreeze_every_k_steps=500, unfreeze_only_k_layers=None):
        self.unfreeze_every_k_steps = unfreeze_every_k_steps
        self.steps = 0
        self.last_unfreeze_candidate = None
        self.name = "encoder"
        self.unfreeze_only_k_layers = unfreeze_only_k_layers

    def freeze_everything(self, pl_module):
        for param in pl_module.siamese_model.parameters():
            param.requires_grad = False

    def unfreeze_layer(self, pl_module, layer_id):
        attr = getattr(pl_module.siamese_model, self.name)
        if hasattr(attr, "transformer"):
            layer_attr = attr.transformer
        if hasattr(attr, "encoder"):
            layer_attr = attr.encoder
        if hasattr(layer_attr, "layer"):
            another_attr = layer_attr.layer
        if hasattr(layer_attr, "albert_layer_groups"):
            another_attr = layer_attr.albert_layer_groups
        for param in another_attr[layer_id].parameters():
            param.requires_grad = True

    def on_batch_start(self, trainer, pl_module):
        attr = getattr(pl_module.siamese_model, self.name)
        if hasattr(attr, "transformer"):
            layer_attr = attr.transformer
        if hasattr(attr, "encoder"):
            layer_attr = attr.encoder
        if self.steps == 0:
            # self.freeze_everything(pl_module)
            self.cnt = 0
            if hasattr(layer_attr, "layer"):
                self.last_unfreeze_candidate = len(layer_attr.layer) - 1

                if self.unfreeze_only_k_layers is not None:
                    self.cnt = self.last_unfreeze_candidate - self.unfreeze_only_k_layers
            elif hasattr(layer_attr, "albert_layer_groups"):
                self.last_unfreeze_candidate = 0
        elif self.steps % self.unfreeze_every_k_steps == 0:
            if self.last_unfreeze_candidate >= self.cnt:
                self.unfreeze_layer(pl_module, self.last_unfreeze_candidate)
                print("unfreeze {} transformer layer".format(self.last_unfreeze_candidate))
                self.last_unfreeze_candidate -= 1
            else:
                if self.unfreeze_only_k_layers is None:
                    for param in attr.parameters():
                        param.requires_grad = True
                    print("unfreeze all")
        self.steps += 1
