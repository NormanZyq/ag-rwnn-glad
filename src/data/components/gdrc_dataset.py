# %%
import os
import random

import dgl
import torch
from dgl.data import DGLDataset
from dgl.data.tu import TUDataset

from src.data.components.p53_dataset import P53Dataset


class GDRCDataset(DGLDataset):
    def __init__(self, name="AIDS",
                 down_sample_label=1,
                 down_sample_rate=0.1,
                 num_sample=5,
                 walk_length=10,
                 re_gen_ds_labels=False,
                 url=None,
                 raw_dir="data/raw",
                 save_dir="data/processed",
                 stage: str = "train"):
        super().__init__(name=name, url=url, raw_dir=raw_dir, save_dir=save_dir)
        self.stage = stage

        if 'p53' in name:
            tu_dataset_delegate = P53Dataset(name, raw_dir)
            tu_dataset_delegate.process()
        else:
            tu_dataset_delegate = TUDataset(name, raw_dir=raw_dir)
        graphs = tu_dataset_delegate.graph_lists
        labels = tu_dataset_delegate.graph_labels

        # load down sampling result if exists
        if not os.path.exists('{}/downsample_ids'.format(raw_dir)):
            os.makedirs('{}/downsample_ids'.format(raw_dir))
        if os.path.exists('{}/downsample_ids/{}_ds_ids.txt'.format(raw_dir, name)) and not re_gen_ds_labels:
            with open('{}/downsample_ids/{}_ds_ids.txt'.format(raw_dir, name), 'r') as f:
                use_ids = eval(f.readline())
        else:
            use_ids = []
            for i in range(len(labels)):
                if labels[i].tolist()[0] == down_sample_label:
                    if random.random() <= down_sample_rate:
                        use_ids.append(i)
                else:
                    use_ids.append(i)
            # save to file
            with open('{}/downsample_ids/{}_ds_ids.txt'.format(raw_dir, name), 'w') as f:
                f.write(str(use_ids))

        # log: building data
        print('building data...')
        self.graphs = []
        self.labels = []
        for idx in use_ids:
            # perform random walk on graph[idx]
            nodes = graphs[idx].nodes()

            for it in range(num_sample):
                # use dgl to impl random walk
                traces, _ = dgl.sampling.random_walk(
                    graphs[idx],
                    nodes,
                    length=walk_length,
                    restart_prob=0.3
                )
                # count the unique nodes in the traces
                # dim of traces is expected to be [num_sample, walk_length]
                # create temp feature
                substructure_node_counting_feat = []
                for tr in traces:
                    substructure_node_counting_feat.append(tr.unique().numel())
                substructure_node_counting_feat = torch.tensor(substructure_node_counting_feat)
                if 'sub_attr' in graphs[idx].ndata:
                    graphs[idx].ndata['sub_attr'] = torch.cat(
                        (graphs[idx].ndata['sub_attr'], substructure_node_counting_feat.unsqueeze(1)),
                        dim=1
                    )
                else:
                    graphs[idx].ndata['sub_attr'] = substructure_node_counting_feat.unsqueeze(1)

            self.graphs.append(graphs[idx].add_self_loop())
            self.labels.append(labels[idx])

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
