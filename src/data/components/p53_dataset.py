import dgl
import numpy as np
import torch
from dgl.data import TUDataset
from dgl.data.utils import loadtxt, load_graphs
from dgl.convert import graph as dgl_graph
from dgl import backend as F
from torch.nn.functional import one_hot
import os

class P53Dataset(TUDataset):
    def __init__(self,
                 name,
                 raw_dir=None,
                 force_reload=False,
                 verbose=False,
                 transform=None, ):
        super().__init__(name, raw_dir, force_reload, verbose, transform)
        self.num_labels = None
        self.graph_labels = None
        self.max_num_node = None
        self.attr_dict = None
        self.graph_lists = None

    def process(self):
        # 这一步把他直接还原回去（假设全都是从1开始的，
        A_txt = loadtxt(super()._file_path("A"), delimiter=",").astype(int)
        DS_edge_list = super()._idx_from_zero(
            A_txt
        )
        DS_indicator = super()._idx_from_zero(
            loadtxt(super()._file_path("graph_indicator"), delimiter=",").astype(
                int
            )
        )

        if os.path.exists(super()._file_path("graph_labels")):
            DS_graph_labels = super()._idx_reset(
                loadtxt(super()._file_path("graph_labels"), delimiter=",").astype(
                    int
                )
            )
            self.num_labels = int(max(DS_graph_labels) + 1)
            self.graph_labels = F.tensor(DS_graph_labels)
        elif os.path.exists(super()._file_path("graph_attributes")):
            DS_graph_labels = loadtxt(
                super()._file_path("graph_attributes"), delimiter=","
            ).astype(float)
            self.num_labels = None
            self.graph_labels = F.tensor(DS_graph_labels)
        else:
            raise Exception("Unknown graph label or graph attributes")

        g = dgl_graph(([], []))
        # 这一步把他直接还原回去（假设全都是从1开始的）（原始的代码是`+1`
        g.add_nodes(int(DS_edge_list.max()) + np.min(A_txt))
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])

        self.attr_dict = {
            "node_labels": ("ndata", "node_labels"),
            "node_attributes": ("ndata", "node_attr"),
            "edge_labels": ("edata", "edge_labels"),
            "edge_attributes": ("edata", "node_labels"),
        }
        for filename, field_name in self.attr_dict.items():
            try:
                data = loadtxt(super()._file_path(filename), delimiter=",")
                if "label" in filename:
                    data = F.tensor(super()._idx_from_zero(data))
                else:
                    data = F.tensor(data)
                if 'node_labels' in filename:
                    # save the raw node_labels as well
                    getattr(g, field_name[0])[field_name[1] + "_raw"] = data
                    # convert to one hot
                    data = one_hot(data.squeeze(), data.max() + 1)
                getattr(g, field_name[0])[field_name[1]] = data
            except IOError:
                pass

        self.graph_lists = [g.subgraph(node_idx) for node_idx in node_idx_list]
        # super().graph_lists = self.graph_lists
        # super().num_labels = self.num_labels
        # super().graph_labels = self.graph_labels
        # super().max_num_node = self.max_num_node
        # super().attr_dict = self.attr_dict



