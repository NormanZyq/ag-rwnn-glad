from typing import Tuple

import dgl
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from dgl.nn.pytorch import GINConv
from torch.nn import Sequential, Linear, ReLU
# from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torchmetrics import AUROC, MeanMetric, MaxMetric


class OCGTL(pl.LightningModule):
    def __init__(self,
                 dim_features,
                 num_trans,
                 dim_targets,
                 num_layers,
                 hidden_dim,
                 norm_layer,
                 bias,
                 aggregation,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool,
                 ):
        super(OCGTL, self).__init__()
        self.save_hyperparameters(logger=False)

        # num_trans = config['num_trans']
        # dim_targets = config['hidden_dim']
        # num_layers = config['num_layers']
        # self.device = config['device']
        self.gins = []
        for _ in range(num_trans):
            self.gins.append(GIN(dim_features,
                                 dim_targets,
                                 hidden_dim,
                                 num_layers,
                                 norm_layer,
                                 bias,
                                 aggregation))
        self.gins = nn.ModuleList(self.gins)
        self.center = nn.Parameter(torch.empty(1, 1, dim_targets * num_layers), requires_grad=True)

        self.criterion = OCGTL_loss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # eval metrics - AUC
        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        self.val_auc_best = MaxMetric()

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.center)
        for nn in self.gins:
            nn.reset_parameters()

    def forward(self, graph, feat):
        # data = data.to(self.device)
        z_cat = []
        for i, model in enumerate(self.gins):
            z = model(graph, feat)
            z_cat.append(z.unsqueeze(1))
        z_cat = torch.cat(z_cat, 1)
        z_cat[:, 0] = z_cat[:, 0] + self.center[:, 0]
        return [z_cat, self.center]

    def model_step(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = graph.ndata['node_attr'].float()
        output = self.forward(graph, feat)
        # loss
        loss = self.criterion(output)
        loss_mean = loss.mean()
        score = loss.detach()

        return loss_mean, score

    def training_step(self, batch, batch_idx):
        graph, labels = batch
        loss, pred = self.model_step(graph)
        labels = labels.squeeze()
        self.train_loss(loss.item())
        self.train_auc(pred, labels)

        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        graph, labels = batch
        loss, pred = self.model_step(graph)
        labels = labels.squeeze()
        self.val_loss(loss.item())
        self.val_auc(pred, labels)

        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        auc = self.val_auc.compute()  # get current val acc
        self.val_auc_best(auc)  # update best so far val acc
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        graph, labels = batch
        loss, pred = self.model_step(graph)
        labels = labels.squeeze()
        self.test_loss(loss.item())
        self.test_auc(pred, labels)

        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}


class OCGTL_loss(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temp = temperature

    def forward(self, z_c, eval=False):

        z = z_c[0]
        c = z_c[1]

        z_norm = (z - c).norm(p=2, dim=-1)
        z = F.normalize(z, p=2, dim=-1)

        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, num_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1
        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp)  # n,k-1

        loss_tensor = torch.pow(z_norm[:, 1:], 1) + (torch.log(trans_matrix) - torch.log(pos_sim))

        if eval:
            score = loss_tensor.sum(1)
            return score
        else:
            loss = loss_tensor.sum(1)
            return loss


class GraphNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super(GraphNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=affine)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(dim), requires_grad=affine)

    def forward(self, node_emb, graph: dgl.DGLGraph):
        num_nodes_list = graph.batch_num_nodes()
        with graph.local_scope():
            graph.ndata['node_emb'] = node_emb
            node_mean = dgl.mean_nodes(graph, 'node_emb')
            # node_mean = scatter_mean(node_emb, index, dim=0, dim_size=graph.batch_size)
            node_mean = node_mean.repeat_interleave(num_nodes_list, 0)

            sub = node_emb - node_mean * self.scale
            graph.ndata['sub_pow2'] = sub.pow(2)
            # node_std = scatter_mean(sub.pow(2), index, dim=0, dim_size=graph.batch_size)
            node_std = dgl.mean_nodes(graph, 'sub_pow2')
            node_std = torch.sqrt(node_std + 1e-8)
            node_std = node_std.repeat_interleave(num_nodes_list, 0)
            norm_node = self.weight * sub / node_std + self.bias
            return norm_node

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)
        init.ones_(self.scale)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, bias=True):
        super(MLP, self).__init__()
        self.lin1 = Linear(in_dim, hidden, bias=bias)
        self.lin2 = Linear(hidden, out_dim, bias=bias)

    def forward(self, z):
        z = self.lin2(F.relu(self.lin1(z)))
        return z

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


class GIN(torch.nn.Module):
    def __init__(self, dim_features,
                 dim_targets,
                 hidden_dim,
                 num_layers,
                 norm_layer,
                 bias,
                 aggregation, ):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.nns = []
        self.convs = []
        self.norms = []
        self.projs = []
        self.use_norm = norm_layer

        if aggregation == 'add':
            self.pooling = global_add_pool
            # dgl.mean_nodes(graph, node_features)
        elif aggregation == 'mean':
            self.pooling = global_mean_pool

        for layer in range(self.num_layers):
            if layer == 0:
                input_emb_dim = dim_features
            else:
                input_emb_dim = hidden_dim
            self.nns.append(Sequential(Linear(input_emb_dim, hidden_dim, bias=bias), ReLU(),
                                       Linear(hidden_dim, hidden_dim, bias=bias)))
            self.convs.append(GINConv(self.nns[-1], learn_eps=bias))  # Eq. 4.2
            if self.use_norm == 'gn':
                self.norms.append(GraphNorm(hidden_dim, True))
            self.projs.append(MLP(hidden_dim, hidden_dim, dim_targets, bias))

        self.nns = nn.ModuleList(self.nns)
        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.projs = nn.ModuleList(self.projs)

    def forward(self, graph, feat):
        # x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        z_cat = []
        with graph.local_scope():
            for layer in range(self.num_layers):
                feat = self.convs[layer](graph, feat)
                if self.use_norm == 'gn':
                    feat = self.norms[layer](feat, graph)
                feat = F.relu(feat)
                z = self.projs[layer](feat)
                graph.ndata['z'] = z
                z = self.pooling(graph, 'z')
                z_cat.append(z)
            z_cat = torch.cat(z_cat, -1)
        return z_cat

    def reset_parameters(self):
        for norm in self.norms:
            norm.reset_parameters()
        # for conv in self.convs:
        #     conv.reset_parameters()
        for proj in self.projs:
            proj.reset_parameters()


def global_add_pool(graphs, node_features):
    pooled_features = dgl.sum_nodes(graphs, node_features)
    return pooled_features


def global_mean_pool(graphs, node_features):
    pooled_features = dgl.mean_nodes(graphs, node_features)
    return pooled_features
