from typing import Any, Dict, Tuple

import dgl
import lightning.pytorch as pl
import torch
from torch.autograd import Variable
from torchmetrics import MeanMetric, AUROC, MaxMetric

from src.models.baseline.gsn.gsn_conv import GSNConv
from src.models.components.graph_classifier import GraphClassifier


class GSN(pl.LightningModule):
    def __init__(self,
                 input_size,
                 hidden_size1,
                 hidden_size2,
                 output_size,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool,
                 str_dropout=0.2,
                 classifier_dropout=0.2
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.single_layer = hidden_size2 <= 0

        if self.single_layer:
            # single-layer gsn
            self.conv1 = GSNConv(input_size, hidden_size1)
            self.graph_classifier = GraphClassifier(hidden_size1, output_size)
        else:
            # two-layer gsn
            self.conv1 = GSNConv(input_size, hidden_size1)
            self.conv2 = GSNConv(hidden_size1, hidden_size2)
            self.graph_classifier = GraphClassifier(hidden_size2, output_size)

        self.dropout = torch.nn.Dropout()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        self.val_auc_best = MaxMetric()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, graph):
        node_attr = graph.ndata['node_attr'].float()
        sub_counting_attr = graph.ndata['counting'].float()

        feat = self.conv1(graph, node_attr, sub_counting_attr)    # in the gsn conv we added relu
        feat = self.dropout(feat)
        # if not self.single_layer:
        #     feat = self.conv2(graph)
        #     feat = self.dropout(feat)
        pred = self.graph_classifier(graph, feat)
        return pred

    def model_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor],):
        # g, feat, center, labels = batch
        graph, label = batch
        pred = self.forward(graph)
        loss = self.criterion(pred, label.float())
        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.train_loss(loss.item())
        self.train_auc(pred, batch[1])
        self.log('train/loss', self.train_loss, on_epoch=True, prog_bar=True)
        self.log('train/auc', self.train_auc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.val_loss(loss.item())
        self.val_auc(pred, batch[1])
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        auc = self.val_auc.compute()  # get current val acc
        self.val_auc_best(auc)  # update best so far val acc
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.test_loss(loss.item())
        self.test_auc(pred, batch[1])
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters(), )
        # torch.optim.SGD()
        return {"optimizer": optimizer}
