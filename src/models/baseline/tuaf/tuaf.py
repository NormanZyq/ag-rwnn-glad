from typing import Any, Dict

import lightning.pytorch as pl
import torch
from torch.autograd import Variable
from torchmetrics import MeanMetric, AUROC, MaxMetric, F1Score


def loss_function(re_triple_adj, tuple_adj, center, h):
    dist, scores = anomaly_score(center, h)
    loss = torch.mean(dist) + torch.mean((re_triple_adj - tuple_adj) ** 2)
    return loss, dist, scores


def anomaly_score(data_center, outputs):
    dist = torch.sum((outputs - data_center) ** 2, dim=1)
    scores = torch.sqrt(dist)
    return dist, scores


def init_center(data, model, eps=0.001):
    outputs = []
    # c = torch.zeros(output_dim)
    model.eval()
    with torch.no_grad():
        for index, g in enumerate(data):
            x = Variable(g['feat_triple'].float(),
                         requires_grad=False)
            tuple_adj = Variable(g['adj'].float(),
                                 requires_grad=False)
            _, h = model(x, tuple_adj)
            outputs.append(torch.mean(h, dim=0))
        if len(outputs) == 1:
            outputs = torch.unsqueeze(outputs[0], 0)
        else:
            outputs = torch.stack(outputs, 0)

        # get the inputs of the batch
        n_samples = outputs.shape[0]
        c = torch.sum(outputs, dim=0)
    c /= n_samples

    # If c_i is too close to 0, set to +-eps.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


class TUAF(pl.LightningModule):
    def __init__(self, model,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # self.model = OutlierModel(feat_dim, hidden_dim, output_dim, dropout)
        self.model = model

        self.train_center = None

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        # self.train_f1 = F1Score(task='binary')
        # self.val_f1 = F1Score(task='binary')
        # self.test_f1 = F1Score(task='binary')

        self.val_auc_best = MaxMetric()

        self.center = None

    def forward(self, data):
        x_tr = Variable(data['feat_triple'].float(),
                        requires_grad=False)
        tuple_adj_tr = Variable(data['adj'].float(),
                                requires_grad=False)
        re_tuple_adj, h_tr = self.model(x_tr, tuple_adj_tr)
        return re_tuple_adj, h_tr, tuple_adj_tr

    def model_step(self, batch):
        # g, feat, center, labels = batch
        data = batch
        re_tuple_adj, h_tr, tuple_adj_tr = self.forward(data)
        loss_tr, dist_tr, score_tr = loss_function(re_tuple_adj, tuple_adj_tr, self.center, h_tr)
        return loss_tr, dist_tr, score_tr

    def training_step(self, batch, batch_idx):
        loss_tr, dist_tr, score_tr = self.model_step(batch)
        target = batch['graph_label'].squeeze() ^ 1
        # target = batch['graph_label'].squeeze()
        self.train_loss(loss_tr.item())
        self.train_auc(score_tr, target)
        # self.train_f1(score_tr, target)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss_tr

    def validation_step(self, batch, batch_idx):
        loss_v, dist_v, score_v = self.model_step(batch)
        self.val_loss(loss_v.item())
        target = batch['graph_label'].squeeze() ^ 1
        # target = batch['graph_label'].squeeze()
        self.val_auc(score_v, target)
        # self.val_f1(score_v, target)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        # self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        auc = self.val_auc.compute()  # get current val acc
        self.val_auc_best(auc)  # update best so far val acc
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss_te, dist_te, score_te = self.model_step(batch)
        self.test_loss(loss_te.item())
        target = batch['graph_label'].squeeze() ^ 1
        # target = batch['graph_label'].squeeze()
        self.test_auc(score_te, target)
        # self.test_f1(score_te, target)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        # self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
