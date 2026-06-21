from typing import Any, Dict, Tuple

import dgl
import lightning.pytorch as pl
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import AUROC, F1Score


class BasicGNN(pl.LightningModule):
    def __init__(
            self,
            graph_conv: torch.nn.Module,
            graph_classifier: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None:
        """
        :param conv: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.net = net
        self.graph_conv = graph_conv
        self.graph_classifier = graph_classifier

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        self.train_f1 = F1Score(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.test_f1 = F1Score(task='binary')

        self.val_auc_best = MaxMetric()

    def forward(self, graph, **kwargs):
        feat = graph.ndata['node_attr'].float()
        forward_out = self.graph_conv(graph, feat)
        forward_out = forward_out.squeeze()
        forward_out = self.graph_classifier(graph, forward_out)

        return forward_out
    
    def concat_attrs(self, g):
        return g.ndata['node_attr'].float()

    def model_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor], ) -> Tuple[
        torch.Tensor, torch.Tensor]:
        g, feat, labels = batch
        labels = labels.float()
        forward_out = self.forward(g, feat)
        loss_forward = self.criterion(forward_out, labels)
        loss = loss_forward
        return loss, forward_out

    def _get_base_data(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx) -> Tuple[
        dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        g, labels = batch
        bn = g.batch_num_nodes()
        g.set_batch_num_nodes(bn)

        feat = g.ndata['node_attr'].float()
        return g, feat, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        g, feat, labels = self._get_base_data(batch, batch_idx)
        loss, forward_out = self.model_step((g, feat, labels))
        self.train_loss(loss.item())
        self.train_auc(forward_out, labels)
        self.train_f1(forward_out, labels)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('train/auc', self.train_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        g, feat, labels = self._get_base_data(batch, batch_idx)
        loss, forward_out = self.model_step((g, feat, labels))
        self.val_loss(loss.item())
        self.val_auc(forward_out, labels)
        self.val_f1(forward_out, labels)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        auc = self.val_auc.compute()
        self.val_auc_best(auc)
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        g, feat, labels = self._get_base_data(batch, batch_idx)
        loss, forward_out = self.model_step((g, feat, labels))
        self.test_loss(loss.item())
        self.test_auc(forward_out, labels)
        self.test_f1(forward_out, labels)

        # logging
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}
