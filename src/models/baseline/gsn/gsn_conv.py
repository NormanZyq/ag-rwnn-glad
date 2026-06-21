import dgl
import torch
import torch.nn as nn
from torch.nn import init


def message_func(edges):
    """cat_ndata_edata_counting"""
    if 'h' in edges.src:
        tmp = torch.cat((edges.src['h'],
                         edges.dst['h'],
                         edges.src['counting'],
                         edges.dst['counting']), dim=1)
    else:
        tmp = torch.cat((edges.src['counting'],
                         edges.dst['counting']), dim=1)
    if 'h' in edges.data:
        tmp = torch.cat((tmp, edges.data['h']), dim=1)
    return {'m': tmp}


def reduce_func(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class GSNConv(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph: dgl.DGLGraph, node_attr, sub_counting):
        # graph, labels = batch
        with graph.local_scope():
            # process feat (ref GraphConv)
            weight = self.weight
            node_attr_src, node_attr_dst, = dgl.utils.expand_as_pair(node_attr, graph)
            sub_counting_src, sub_counting_dst = dgl.utils.expand_as_pair(sub_counting, graph)
            # todo 怎么才是满足GSN文章要求的
            # norm
            # degs = graph.out_degrees().to(node_attr_src).clamp(min=1)
            # norm = torch.pow(degs, -0.5)
            # shp = norm.shape + (1,) * (node_attr_src.dim() - 1)
            # norm = torch.reshape(norm, shp)
            # node_attr_src = node_attr_src * norm
            # apply h
            graph.srcdata["h"] = node_attr_src
            graph.dstdata['h'] = node_attr_dst
            graph.srcdata['counting'] = sub_counting_src
            graph.dstdata['counting'] = sub_counting_dst
            # update all
            graph.update_all(message_func=message_func, reduce_func=reduce_func)
            rst = graph.dstdata['h']
            rst = torch.matmul(rst, weight)
            # norm (ref GraphConv)
            degs = graph.in_degrees().to(node_attr_dst).clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (node_attr_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm
            # bias
            rst = rst + self.bias
            # act
            rst = rst.relu()
            return rst


if __name__ == '__main__':
    g = dgl.rand_graph(33, 50)
    # g = dgl.DGLGraph(g)
    g.ndata['h'] = torch.randn((33, 16))
    g.ndata['counting'] = torch.randn((33, 11))
    g.edata['h'] = torch.randn((50, 8))
    gsn_conv = GSNConv(16 * 2 + 11 * 2 + 8, 64)
    rst = gsn_conv(g)
    print(rst)
