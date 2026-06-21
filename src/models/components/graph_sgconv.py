import torch
from dgl.nn.pytorch.conv import SGConv
from torch.nn import Linear


class MySGConv(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv = SGConv(input_size, output_size)
        self.linear = Linear(hidden_size, output_size)

    def forward(self, g, feat):
        feat = self.conv(g, feat)
        feat = feat.relu()
        # feat = F.dropout(feat, p=0.2)
        feat = self.linear(feat)
        feat = feat.relu()

        return feat
