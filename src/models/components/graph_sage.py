import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv


class MyGraphSage(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, aggregator_type='gcn', dropout=0):
        super().__init__()
        input_size = eval(input_size) if isinstance(input_size, str) else input_size
        output_size = eval(output_size) if isinstance(output_size, str) else output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv = SAGEConv(input_size, output_size, aggregator_type, feat_drop=dropout)

    def forward(self, g, feat):
        feat = self.conv(g, feat)
        feat = F.leaky_relu(feat)

        return feat
