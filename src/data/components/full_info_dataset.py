# %%
import random

import torch
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.data.tu import TUDataset

from src.data.components.p53_dataset import P53Dataset


class FullInfoDataset(DGLDataset):
    def __init__(self, name="AIDS",
                 down_sample_label=1,
                 down_sample_rate=0.1,
                 default_feat_dim=-1,
                 re_gen_ds_labels=False,
                 url=None,
                 raw_dir="data/raw",
                 save_dir="data/processed",
                 stage: str = "train",
                 xor=False,
                 seed=12345):
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
        requires_empty_node_attr = False
        if 'node_attr' not in graphs[0].ndata.keys():
            if default_feat_dim > 0:
                # if there are no node attributes, then initialize them with random tensors
                print('No node attr found. Generating initial embeddings with dim=', default_feat_dim)
                requires_random_feat = True
            else:
                requires_empty_node_attr = True
        # filter the samples
        self.graphs = []
        self.labels = []
        num_node_labels = 0
        for idx in use_ids:
            if requires_empty_node_attr:
                graphs[idx].ndata['node_attr'] = torch.zeros((graphs[idx].num_nodes(), 0))
            if requires_random_feat:
                # generate trainable embeddings
                node_features = torch.nn.Embedding(graphs[idx].num_nodes(), default_feat_dim, dtype=torch.float64)
                graphs[idx].ndata['node_attr'] = node_features.weight
            if 'node_labels' in graphs[idx].ndata:
                num_node_labels = max(graphs[idx].ndata['node_labels'].max(), num_node_labels)
            self.graphs.append(graphs[idx].add_self_loop())
            self.labels.append(labels[idx] ^ 1 if xor else labels[idx])

        num_node_labels += 1
        for idx in range(len(self.graphs)):
            # if node labels exist in the data
            if 'node_labels' in self.graphs[0].ndata:
                # 对于非tox的数据，重新遍历一次graphs，将node_labels转换为one hot 存进去
                if 'tox' not in name.lower():
                    self.graphs[idx].ndata['node_labels'] = F.one_hot(self.graphs[idx].ndata['node_labels'].squeeze(),
                                                                      num_classes=num_node_labels)
                # for all data including tox, cat the node attr with node labels
                self.graphs[idx].ndata['node_attr'] = torch.cat((self.graphs[idx].ndata['node_attr'],
                                                                 self.graphs[idx].ndata['node_labels']), dim=1)

        random.seed(seed)
        print('seed has been reset to your config,', seed)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
