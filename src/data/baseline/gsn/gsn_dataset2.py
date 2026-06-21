import random

import torch
from dgl.data.tu import TUDataset

from src.data.components.p53_dataset import P53Dataset
from src.utils.baseline.gsn_utils.utils import dgl_graph2pyg, subgraph_isomorphism_vertex_counts, automorphism_orbits, \
    pad_tensors
from src.utils.utils import gen_use_ids, down_sampling
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class GSNDataset(torch.utils.data.Dataset):
    def __init__(self, name="AIDS",
                 down_sample_label=0,
                 down_sample_rate=0.1,
                 default_feat_dim=-1,
                 re_gen_ds_labels=False,
                 url=None,
                 raw_dir="data/raw",
                 save_dir="data/processed",
                 # subgraph_dict=None,
                 induced=False,
                 is_directed=False,
                 stage: str = "train", **kwargs):
        super().__init__()
        # graphs, num_classes = load_data(data_path, dataset_name, False)
        # if subgraph_dict is None:
        #     subgraph_dict = {}
        if 'tox' in name.lower():
            tu_dataset_delegate = P53Dataset(name, raw_dir)
            tu_dataset_delegate.process()
        else:
            tu_dataset_delegate = TUDataset(name, raw_dir=raw_dir)
        tu_graphs = tu_dataset_delegate.graph_lists
        tu_labels = tu_dataset_delegate.graph_labels

        self.cfg_seed = kwargs['seed']
        random.seed(12345)  # we fix seed to 12345 for data generation and processing
        print('seed has been fixed to', 12345)

        use_ids = []
        for i in range(len(tu_labels)):
            if tu_labels[i].tolist()[0] == down_sample_label:
                if random.random() <= down_sample_rate:
                    use_ids.append(i)
            else:
                use_ids.append(i)

        graphs, labels, = down_sampling(tu_graphs, tu_labels, use_ids)

        # subgraph isomorphism counting
        counting_tensor_list = []
        print('Calculating isomorphism subgraph count...')
        for graph in tqdm(graphs):
        # for graph in graphs:
            edge_index, _ = dgl_graph2pyg(graph)
            num_nodes = graph.num_nodes()
            edge_list = torch.stack(graph.edges()).T
            if len(edge_list) == 0:
                # Note: a bug in the official code?
                #       if the graph contains only discrete nodes without an edge
                #       the get_custom_edge_listfollowing counting method will raise an error
                #       that tells you cannot calculate isomorphism across empty graphs
                # ------
                # So we make a zero tensor directly
                counting = torch.zeros((num_nodes, 1))
            else:
                subgraph, orbit_partition, orbit_membership, aut_count = automorphism_orbits(edge_list=edge_list,
                                                                                             directed=False,
                                                                                             directed_orbits=False)

                subgraph_dict = {'subgraph': subgraph, 'orbit_partition': orbit_partition,
                                 'orbit_membership': orbit_membership, 'aut_count': aut_count}
                counting = subgraph_isomorphism_vertex_counts(edge_index,
                                                              subgraph_dict,
                                                              induced,
                                                              num_nodes,
                                                              is_directed)
                counting_tensor_list.append(counting)
            graph.ndata['counting'] = counting
        print('Finish!')

        # pad the counting tensors
        padded_counting_list = pad_tensors(counting_tensor_list, 0)
        for g, counting in zip(graphs, padded_counting_list):
            g.ndata['counting'] = counting

        self.labels = labels
        self.num_classes = tu_dataset_delegate.num_classes
        self.graphs = graphs

        random.seed(self.cfg_seed)
        print('seed has been reset to your config, ', self.cfg_seed)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = GSNDataset()
    print(dataset)
