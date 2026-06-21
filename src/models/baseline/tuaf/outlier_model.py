import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, has_bias=False):
        super(GCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.FloatTensor(
            input_dim, output_dim))
        if has_bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(
            self.weights, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, adj):
        h_w = torch.matmul(x, self.weights)
        output = torch.bmm(adj, h_w)
        if self.bias is not None:
            return output + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, bias=True):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # middle layer dim
        self.output_dim = output_dim
        self.gcn1 = GCNConv(input_dim, hidden_dim, has_bias=bias)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim, has_bias=bias)
        self.gcn3 = GCNConv(hidden_dim, output_dim, has_bias=bias)
        self.dropout_prob = dropout_prob
        self.act = nn.Sigmoid()

    def forward(self, x, adj):
        # layer-1
        h1 = self.gcn1(x, adj)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, self.dropout_prob, training=self.training)
        # layer-2
        h2 = self.gcn2(h1, adj)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, self.dropout_prob, training=self.training)
        # layer-3
        h3 = self.gcn3(h2, adj)
        return h3


class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, h):
        h_t = h.transpose(1, 2)
        adj_tmp = torch.bmm(h, h_t)
        re_adj = self.act(adj_tmp)
        return re_adj


class FusionReadout(nn.Module):
    def __init__(self, out_dim):
        super(FusionReadout, self).__init__()
        self.weights_1 = nn.Parameter(torch.FloatTensor(
            out_dim, out_dim))
        self.weights_2 = nn.Parameter(torch.FloatTensor(
            out_dim, out_dim))
        self.emb_layer = nn.Linear(out_dim, 1)

        self.act_softmax = nn.Softmax(dim=-1)

        self.act_tanh = nn.Tanh()
        self.act_leakyrelu = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(
            self.weights_1, mode='fan_out', nonlinearity='relu')
        init.kaiming_uniform_(
            self.weights_2, mode='fan_out', nonlinearity='relu')

    def forward(self, h_triple):
        h_mean = torch.mean(h_triple, dim=1)
        h_mean_ = h_mean.repeat(h_triple.shape[1], 1).reshape(
            h_mean.shape[0], h_triple.shape[1], h_mean.shape[1])
        tmp1 = torch.matmul(h_triple, self.weights_1)
        tmp2 = torch.matmul(h_mean_, self.weights_2)

        # compute coeffcients
        h_assign = torch.mul(tmp1, tmp2)
        coeffcient = self.emb_layer(h_assign)
        coeffcient = self.act_leakyrelu(coeffcient)
        coeffcient = self.act_softmax(coeffcient)

        h_g = torch.mul(h_triple, coeffcient)

        # fusion_sum
        h_g = torch.sum(h_g, dim=1)
        h_g = self.act_tanh(h_g)
        return h_g


class OutlierModel(nn.Module):
    def __init__(self,
                 feat_dim,
                 hidden_dim,
                 out_dim,
                 dropout_prob):
        super().__init__()
        self.learning_triple = GCN(feat_dim, hidden_dim, out_dim, dropout_prob)
        self.triple_updating = Reconstruction()
        self.graph_readout = FusionReadout(out_dim)

    def forward(self, x_triple, tuple_adj):
        h_triple = self.learning_triple(x_triple, tuple_adj)
        re_tuple_adj = self.triple_updating(h_triple)
        h_g = self.graph_readout(h_triple)

        return re_tuple_adj, h_g
