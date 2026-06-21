import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv


class MyGraphSage(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, aggregator_type='gcn', dropout=0):
        super().__init__()
        input_size = eval(input_size) if isinstance(input_size, str) else input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # First layer
        self.conv1 = SAGEConv(input_size, hidden_size, aggregator_type, feat_drop=dropout)
        # Second layer
        self.conv2 = SAGEConv(hidden_size, output_size, aggregator_type, feat_drop=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, g, feat):
        # First layer
        feat = self.conv1(g, feat)
        feat = F.leaky_relu(feat)
        feat = self.dropout(feat)
        
        # Second layer
        feat = self.conv2(g, feat)
        feat = F.leaky_relu(feat)
        
        return feat