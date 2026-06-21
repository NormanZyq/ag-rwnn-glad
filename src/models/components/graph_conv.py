import torch
from dgl.nn.pytorch.conv import GraphConv


class MyGraphConv(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super().__init__()
        input_size = eval(input_size) if isinstance(input_size, str) else input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = GraphConv(input_size, hidden_size)
        self.conv2 = GraphConv(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, g, feat):
        feat = self.conv1(g, feat)
        feat = self.dropout(feat)
        feat = feat.relu()
        # dgl note: I can put the relu into the definition of `conv1`, in this way DGL will call relu()
        feat = self.conv2(g, feat)  # [batch, N, hidden_size]

        return feat
