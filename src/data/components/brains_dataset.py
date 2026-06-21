import dgl
import numpy as np
import torch
from dgl.data import TUDataset, DGLDataset
from dgl.data.utils import loadtxt, load_graphs
from dgl.convert import graph as dgl_graph
from dgl import backend as F
from torch.nn.functional import one_hot
import os


def gen_graphs(records):
    """
    from GmapAD
    """
    node_name = []
    # Get all node names and assign a global ID to each node
    for i in range(len(records)):
        if records[i] != "\n":
            if 'n' in records[i]:
                n_name = records[i].split(' ')[2].strip('\n')
                if n_name not in node_name:
                    node_name.append(n_name)

    node_index = dict()
    for g_id, name in enumerate(node_name):
        node_index.update({
            name: g_id})

    graph_names = []
    graphs = dict()
    graph = dict()
    nodes = dict()
    edges = dict()
    graph_num = 1
    edge_id = 1

    for i in range(len(records)):
        if records[i] != "\n":
            indicator = records[i].split(' ')[0].strip('\n')

            # Graph ID
            if 'g' == indicator:
                graph_name = records[i].split(' ')[2].strip('\n')
                graph_names.append(graph_name)

            # Node name and ID
            if 'n' == indicator:
                node_id_local = records[i].split(' ')[1].strip('\n')
                n_name = records[i].split(' ')[2].strip('\n')
                nodes.update({
                    node_id_local: n_name
                })

            if 'e' == indicator:
                st_node_id_local = records[i].split(' ')[1].strip('\n')
                ed_node_id_local = records[i].split(' ')[2].strip('\n')
                w = records[i].split(' ')[3].strip('\n')
                edge = [st_node_id_local, ed_node_id_local, w]
                edges.update({
                    edge_id: edge
                })
                edge_id = edge_id + 1

            if 'x' == indicator:
                label = records[i].split(' ')[1].strip('\n')
                if label == '1':
                    g_label = 1
                else:
                    g_label = 0
        else:
            graph.update({
                "nodes": nodes,
                "edges": edges,
                "label": g_label
            })
            graphs.update({
                graph_name: graph
            })
            graph = dict()
            nodes = dict()
            edges = dict()
            edge_id = 0
            graph_num = graph_num + 1

    return graphs, node_index


class BrainsDataset(DGLDataset):
    def __init__(self,
                 name,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 hash_key=(),
                 force_reload=False,
                 verbose=False,
                 transform=None,
                 ):
        super().__init__(name,
                         url=url,
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         hash_key=hash_key,
                         force_reload=force_reload,
                         verbose=verbose,
                         transform=transform, )
        # self.num_labels = None
        self.graph_labels = None
        # self.max_num_node = None
        # self.attr_dict = None
        self.graph_lists = None
        # self.raw_dir = raw_dir
        # self.name = name

    def process(self):
        with open(os.path.join(self.raw_dir, 'Brains', self.name + '.nel')) as f:
            raw_data = f.readlines()
        graphs_tmp, node_index = gen_graphs(raw_data)
        graphs = []
        labels = []
        for graph in graphs_tmp:
            nodes = graphs_tmp[graph]['nodes']
            edges = graphs_tmp[graph]['edges']
            label = graphs_tmp[graph]['label']
            label = np.array([label])
            label = torch.tensor(label, dtype=torch.long)
            st_nodes = []
            ed_nodes = []
            for edge in edges:
                st_nodes.append(node_index[nodes[edges[edge][0]]])
                ed_nodes.append(node_index[nodes[edges[edge][1]]])
            selflinks = list(range(0, len(node_index)))
            st_nodes = st_nodes + selflinks
            ed_nodes = ed_nodes + selflinks

            st_nodes = torch.tensor(st_nodes, dtype=torch.long)
            ed_nodes = torch.tensor(ed_nodes, dtype=torch.long)
            g = dgl.graph((st_nodes, ed_nodes))

            graphs.append(g)
            labels.append(label)

        self.graph_lists = graphs
        self.graph_labels = labels


if __name__ == '__main__':
    dataset = BrainsDataset('KKI', raw_dir='data/raw/Brains')
    dataset.process()
    print(dataset)
