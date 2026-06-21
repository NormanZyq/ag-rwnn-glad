from typing import Any, Dict

import lightning.pytorch as pl
import torch
from torch.autograd import Variable
from torchmetrics import MeanMetric, AUROC, MaxMetric

class GSN(pl.LightningModule):
    def __init__(self,
                 model,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # self.model = OutlierModel(feat_dim, hidden_dim, output_dim, dropout)
        self.model = model

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        self.val_auc_best = MaxMetric()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        prediction = self.model(data, False)
        return prediction

    def model_step(self, batch):
        # g, feat, center, labels = batch
        data = batch
        label = data.y
        prediction = self.forward(data)
        loss = self.criterion(prediction, label)
        return loss, prediction

    def training_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.train_loss(loss.item())
        self.train_auc(pred, batch.y)
        self.log('train/loss', self.train_loss, on_epoch=True, prog_bar=True)
        self.log('train/auc', self.train_auc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.val_loss(loss.item())
        self.val_auc(pred, batch.y)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        auc = self.val_auc.compute()  # get current val acc
        self.val_auc_best(auc)  # update best so far val acc
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.test_loss(loss.item())
        self.test_auc(pred, batch.y)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters(), )
        # torch.optim.SGD()
        return {"optimizer": optimizer}
