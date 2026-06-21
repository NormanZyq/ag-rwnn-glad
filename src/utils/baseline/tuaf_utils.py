import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric


# class TUAFAuc(Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state('auc', default=torch.Tensor(0.), dist_reduce_fx='mean')

# nx.edges()
def convert_to_triple_graph(g):
    adj = np.array(nx.to_numpy_array(g))
    node_o_num = len(adj)
    adj_tri_upper = np.triu(adj, k=0)
    node_o_pairs = np.argwhere(adj_tri_upper == 1)
    tuple_list = [str(i[0]) + '_' + str(i[1]) for i in node_o_pairs]
    node_t_index_list = np.arange(1, len(tuple_list) + 1)
    t_o_node_dict = zip(tuple_list, node_t_index_list)
    t_o_node_dict = dict(t_o_node_dict)

    # construct a new adjacent matrix
    subgraph_list = []
    adj_t = np.zeros((len(tuple_list), len(tuple_list)))
    for i in range(1, node_o_num + 1):
        sub_node_o = []
        for item in tuple_list:
            tmp = item.split("_")
            tmp1 = tmp[0]
            tmp2 = tmp[1]
            if (str(i - 1) == str(tmp1)) | (str(i - 1) == str(tmp2)):
                sub_node_o.append(t_o_node_dict[item] - 1)
        if len(sub_node_o) > 1:
            subgraph_list.append(sub_node_o)
            sub_node_o = np.array(sub_node_o)
            rows = [[i] * len(sub_node_o) for i in sub_node_o]
            cols = [[sub_node_o] * len(sub_node_o)]
            adj_t[rows, cols] = 1
    row, col = np.diag_indices_from(adj_t)
    adj_t[row, col] = 0

    # convert to triple-uint graph
    G_1 = nx.from_numpy_array(adj_t)
    for k in range(0, len(G_1.nodes)):
        tuple_node_name = list(t_o_node_dict.keys())[
            list(t_o_node_dict.values()).index(k + 1)]
        node_tuple = tuple_node_name.split("_")
        index1 = node_tuple[0]
        index2 = node_tuple[1]
        tuple_node1 = g.nodes[int(index1)]['node_labels']
        tuple_node2 = g.nodes[int(index2)]['node_labels']
        # g.edges[[int(index1), int(index2)]]['edge_labels'] = F.one_hot
        tuple_edge_label = g.edges[[int(index1), int(index2)]]['edge_labels']
        # if len(tuple_edge_label)
        G_1.nodes[k]['feat_triple'] = np.concatenate(
            (tuple_node1.numpy(), tuple_edge_label.numpy(), tuple_node2.numpy()), axis=0)

    G_1.graph['label'] = g.graph['label']
    return G_1
