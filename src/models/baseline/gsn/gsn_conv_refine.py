import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GINConv, GraphConv


class GSNConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.gin_conv = GINConv()

    def forward(self, graph: dgl.DGLGraph):
        node_attr = graph['node_attr']


