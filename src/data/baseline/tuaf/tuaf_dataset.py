# %%
import os
import random

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.data.tu import TUDataset

from src.data.components.p53_dataset import P53Dataset
from src.utils.baseline.tuaf_utils import convert_to_triple_graph


class GraphSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, normalize=False, max_num_triples=0):
        self.adj_all = []
        self.feat_triple_all = []
        self.graph_label_all = []
        self.max_num_triples = max_num_triples
        self.feat_triple_dim = len(G_list[0].nodes[0]['feat_triple'])

        for G in G_list:
            adj = np.array(nx.to_numpy_array(G))
            if normalize:
                I = np.eye(adj.shape[0])
                A_hat = adj + I
                D_hat = np.sum(A_hat, axis=0)
                d_hat = np.diag(np.power(D_hat, -0.5).flatten())
                norm_adj = d_hat.dot(A_hat).dot(d_hat)

            self.adj_all.append(norm_adj)
            self.graph_label_all.append(G.graph['label'])

            feat_label_list = np.zeros(
                (self.max_num_triples, self.feat_triple_dim), dtype=float)
            for i_f, u_f in enumerate(G.nodes()):
                feat_label_list[i_f, :] = G.nodes[u_f]['feat_triple']
            self.feat_triple_all.append(feat_label_list)

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_triples = adj.shape[0]
        if self.max_num_triples > num_triples:
            adj_padded = np.zeros((self.max_num_triples, self.max_num_triples))
            adj_padded[:num_triples, :num_triples] = adj
        else:
            adj_padded = adj
        return {'adj': adj_padded,  # adjacent matrix of triple-unit graph
                'feat_triple': self.feat_triple_all[idx],  # triple features
                'graph_label': self.graph_label_all[idx],  # graph label
                'num_triples': num_triples,  # the number of triples
                }


class TUAFDataset(DGLDataset):
    def __init__(self, name="AIDS", down_sample_label=1, down_sample_rate=0.1, default_feat_dim=-1,
                 re_gen_ds_labels=False, url=None,
                 raw_dir="data/raw", save_dir="data/processed", stage: str = "train", seed=12345):
        super().__init__(name=name, url=url, raw_dir=raw_dir, save_dir=save_dir)
        self.stage = stage
        is_tox_dataset = False
        if 'tox' in name.lower():
            is_tox_dataset = True
            tu_dataset_delegate = P53Dataset(name, raw_dir=raw_dir)
            tu_dataset_delegate.process()
        else:
            tu_dataset_delegate = TUDataset(name, raw_dir=raw_dir)
        tu_graphs = tu_dataset_delegate.graph_lists
        tu_graph_labels = tu_dataset_delegate.graph_labels
        # node_labels = tu_dataset_delegate.attg

        random.seed(12345)  # we fix seed to 12345 for data generation and processing
        print('seed has been fixed to', 12345)

        use_ids = []
        for i in range(len(tu_graph_labels)):
            if tu_graph_labels[i].tolist()[0] == down_sample_label:
                if random.random() <= down_sample_rate:
                    use_ids.append(i)
            else:
                use_ids.append(i)

        requires_random_feat = False
        if 'node_attr' not in tu_graphs[0].ndata.keys() or default_feat_dim != -1:
            # test not sure about this operation
            # if there are no node attributes, then initialize them
            requires_random_feat = True
        # filter the samples
        graphs = []
        self.labels = []
        if requires_random_feat:
            print('Generating initial embeddings')
        for idx in use_ids:
            if requires_random_feat:
                # generate trainable embeddings
                # node_features = torch.nn.Parameter(
                #     init.xavier_uniform_(torch.empty(graphs[idx].num_nodes(), default_feat_dim, dtype=torch.float64)))
                node_features = torch.nn.Embedding(tu_graphs[idx].num_nodes(), default_feat_dim, dtype=torch.float64)
                tu_graphs[idx].ndata['node_attr'] = node_features.weight
            graphs.append(tu_graphs[idx])
            self.labels.append(tu_graph_labels[idx])
        # --------------- end filtering graphs --------------

        nx_graphs = []
        num_node_labels = 0
        num_edge_labels = 0
        for i in range(len(graphs)):
            g = graphs[i]
            # nxg = nx.Graph(dgl.to_networkx(g, g.ndata.keys(), g.edata.keys()).to_undirected())
            # nxg.graph['label'] = self.labels[i]
            # nx_graphs.append(nxg)
            num_node_labels = max(num_node_labels, g.ndata['node_labels'].max())
            # if len(g.edata['edge_labels']) == 0:
            #     g.edata['edge_labels'][0] = 0
            # num_edge_labels = max(num_edge_labels, g.edata['edge_labels'].max()) if 'edge_labels' in g.edata else 0
            num_edge_labels = max(num_edge_labels, g.edata['edge_labels'].max() if g.num_edges() != 0 else 0) if 'edge_labels' in g.edata else 0
            # num_node_labels = max(max([nxg.nodes[u]['node_labels'][0].tolist() for u in nxg.nodes]), num_node_labels)

        num_node_labels += 1  # 从0开始的所以+1
        num_edge_labels += 1

        for i in range(len(graphs)):
            g = graphs[i]
            if 'edge_labels' not in g.edata or g.edata['edge_labels'].shape[0] == 0:
                tmp_el = torch.zeros((g.num_edges(), 1))
                g.edata['edge_labels'] = tmp_el
            else:
                g.edata['edge_labels'] = F.one_hot(g.edata['edge_labels'], num_classes=num_edge_labels).squeeze()

            # 如果是tox数据集，那么此时他们的node label就已经是one hot的了不需要再次转换
            if not is_tox_dataset:
                g.ndata['node_labels'] = F.one_hot(g.ndata['node_labels'], num_classes=num_node_labels).squeeze()

        # for g in nx_graphs:
        # for index, u in enumerate(g.nodes):
        #     node_one_hot_label = F.one_hot(g.nodes[u]['node_labels'], num_classes=num_node_labels)
        #     g.nodes[u]['node_labels'] = node_one_hot_label

        for i in range(len(graphs)):
            g = graphs[i]
            nxg = nx.Graph(dgl.to_networkx(g, g.ndata.keys(), g.edata.keys()))
            nxg.graph['label'] = self.labels[i]
            nx_graphs.append(nxg)

        self.triple_graphs = [convert_to_triple_graph(g) for g in nx_graphs]
        self.max_nodes_num = max([G.number_of_nodes() for G in self.triple_graphs])
        self.max_edge_num = max([G.number_of_edges() for G in self.triple_graphs])
        self.max_triples_num = max([G.number_of_nodes() for G in self.triple_graphs])

        random.seed(seed)
        print('seed has been reset to your config,', seed)

    def __getitem__(self, idx):
        return self.triple_graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.triple_graphs)
