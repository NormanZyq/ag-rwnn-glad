# %%
import random

import torch
import torch.nn.init as init
from dgl.data import DGLDataset
from dgl.data.tu import TUDataset
import os

from src.data.components.p53_dataset import P53Dataset


class GDRDataset(DGLDataset):
    def __init__(self, name="AIDS", down_sample_label=1, down_sample_rate=0.1, default_feat_dim=-1,
                 re_gen_ds_labels=False, url=None,
                 raw_dir="data/raw", save_dir="data/processed", stage: str = "train", seed=12345):
        super().__init__(name=name, url=url, raw_dir=raw_dir, save_dir=save_dir)
        self.stage = stage
        if 'tox' in name.lower():
            tu_dataset_delegate = P53Dataset(name, raw_dir=raw_dir)
            tu_dataset_delegate.process()
        else:
            tu_dataset_delegate = TUDataset(name, raw_dir=raw_dir)
        graphs = tu_dataset_delegate.graph_lists
        labels = tu_dataset_delegate.graph_labels

        random.seed(12345)
        print('seed has been fixed to', 12345)

        use_ids = []
        for i in range(len(labels)):
            if labels[i].tolist()[0] == down_sample_label:
                if random.random() <= down_sample_rate:
                    use_ids.append(i)
            else:
                use_ids.append(i)

        requires_random_feat = False
        if 'node_attr' not in graphs[0].ndata.keys() or default_feat_dim != -1:
            # test not sure about this operation
            # if there are no node attributes, then initialize them with random tensors
            print('No node attr found. Generating initial embeddings with dim=', default_feat_dim)
            requires_random_feat = True
        # filter the samples
        self.graphs = []
        self.labels = []
        for idx in use_ids:
            if requires_random_feat:
                # generate trainable embeddings
                # node_features = torch.nn.Parameter(
                #     init.xavier_uniform_(torch.empty(graphs[idx].num_nodes(), default_feat_dim, dtype=torch.float64)))
                node_features = torch.nn.Embedding(graphs[idx].num_nodes(), default_feat_dim, dtype=torch.float64)
                graphs[idx].ndata['node_attr'] = node_features.weight
            self.graphs.append(graphs[idx].add_self_loop())
            self.labels.append(labels[idx])

        random.seed(seed)
        print('seed has been reset to your config,', seed)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
