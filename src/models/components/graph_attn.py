import dgl
import torch
from dgl.nn.pytorch.conv import GATConv
from torch.nn import Linear


class MyGraphAttn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super().__init__()
        input_size = eval(input_size) if isinstance(input_size, str) else input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        num_heads = 2
        # self.N = num_nodes
        self.gat = GATConv(input_size, hidden_size, num_heads=num_heads)
        self.num_heads = 2
        self.linear = Linear(hidden_size * num_heads, output_size)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        # g_sub, g = g
        # feat_sub, feat = feat
        feat = self.gat(g, feat)
        feat = feat.reshape(feat.shape[0], -1)
        feat = feat.relu()
        feat = self.linear(feat)
        feat = feat.relu()

        return feat
