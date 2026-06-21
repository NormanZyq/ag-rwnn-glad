import random
from typing import Tuple

import dgl
import torch


def sample_substructure(graphs_origin, num_sample, num_sub_nodes, sampling_strategy) -> Tuple[
    dgl.DGLGraph, dgl.DGLGraph]:
    if num_sample > 0:
        if sampling_strategy == 'single_node_multiple_sampling':
            sampled_g = single_node_multiple_sampling(graphs_origin, num_sample, num_sub_nodes)
        else:
            sampled_g = graphs_origin
    else:
        sampled_g = graphs_origin
    # add self loop for the original graphs
    unbatched_g = dgl.unbatch(graphs_origin)
    for i in range(len(unbatched_g)):
        unbatched_g[i] = unbatched_g[i].add_self_loop()
    batched_g_origin = dgl.batch(unbatched_g)
    if num_sample <= 0:
        sampled_g = batched_g_origin
    return batched_g_origin, sampled_g


def single_node_multiple_sampling(g, num_sample, num_sub_nodes) -> dgl.DGLGraph:
    """
    从度最小的节点开始重复游走得到不同子图
    """
    batched_g = g
    sampled_g = []
    for g in dgl.unbatch(batched_g):
        degrees = g.in_degrees()
        min_degree = int(torch.min(degrees).tolist())
        indices = torch.nonzero(degrees == min_degree, as_tuple=False)
        start_from = torch.LongTensor([random.choice(indices)] * num_sample)
        # random walk
        traces, _ = dgl.sampling.random_walk(
            g,
            start_from.to(degrees.device),
            length=num_sub_nodes - 1,
        )
        for nodes in traces:
            sampled_g.append(dgl.node_subgraph(g, nodes[nodes.ne(-1)], relabel_nodes=False).add_self_loop())
    sampled_g = dgl.batch(sampled_g)
    return sampled_g
