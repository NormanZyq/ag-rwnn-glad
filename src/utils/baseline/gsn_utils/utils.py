from typing import Optional

import dgl
import graph_tool as gt
import graph_tool.topology as gt_topology
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F



# Now re-implement torch_geometrics's degree function without that package:
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def dgl_graph2pyg(dgl_graph: dgl.DGLGraph):
    """Convert DGL graph to PyG graph.
    """
    edge_index = torch.stack(dgl_graph.edges())
    node_attr = dgl_graph.ndata['node_attr'] if 'node_attr' in dgl_graph.ndata else None
    # create pyg data
    # pyg_graph = Data(x=node_attr, edge_index=edge_index)
    return edge_index, node_attr


def subgraph_isomorphism_vertex_counts(edge_index,
                                       subgraph_dict,
                                       induced,
                                       num_nodes,
                                       is_directed=False):
    directed = is_directed

    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index.transpose(1, 0).cpu().numpy()))
    gt.generation.remove_self_loops(G_gt)
    gt.generation.remove_parallel_edges(G_gt)

    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'],
                                               G_gt,
                                               induced=induced,
                                               subgraph=True,
                                               generator=True)

    ## num_nodes should be explicitly set for the following edge case:
    ## when there is an isolated vertex whose index is larger
    ## than the maximum available index in the edge_index

    counts = np.zeros((num_nodes, len(subgraph_dict['orbit_partition'])))
    for sub_iso_curr in sub_iso:
        for i, node in enumerate(sub_iso_curr):
            # increase the count for each orbit
            counts[node, subgraph_dict['orbit_membership'][i]] += 1
    counts = counts / subgraph_dict['aut_count']

    counts = torch.tensor(counts)

    return counts


def automorphism_orbits(edge_list, print_msgs=False, **kwargs):
    ##### vertex automorphism orbits #####

    directed = kwargs['directed'] if 'directed' in kwargs else False

    graph = gt.Graph(directed=directed)
    gt.Graph()
    graph.add_edge_list(edge_list)
    gt.generation.remove_self_loops(graph)
    gt.generation.remove_parallel_edges(graph)

    # compute the vertex automorphism group
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v

    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role

    orbit_membership_list = [[], []]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse=True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i, vertex in enumerate(orbit_membership_list[0])}

    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit] + [vertex]

    aut_count = len(aut_group)

    if print_msgs:
        print('Orbit partition of given substructure: {}'.format(orbit_partition))
        print('Number of orbits: {}'.format(len(orbit_partition)))
        print('Automorphism count: {}'.format(aut_count))

    return graph, orbit_partition, orbit_membership, aut_count


def pad_tensors(tensors, pad_value=0):
    """
    为了将他count出来的结果第1维度长度一致，只能进行padding了
    """
    # 确定第一维的最大长度
    max_len = max(tensor.size(1) for tensor in tensors)

    # pad每个tensor到最大长度
    padded_tensors = []
    for tensor in tensors:
        # 计算需要在每侧填充的长度
        padding = (0, max_len - tensor.size(1))
        # 在第一维进行padding
        padded_tensor = F.pad(tensor, pad=(0, max_len - tensor.size(1)), value=pad_value)
        padded_tensors.append(padded_tensor)

    return padded_tensors
