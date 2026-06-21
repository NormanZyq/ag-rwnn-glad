import random
from collections import defaultdict
from typing import List, Dict

import dgl
import torch
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.data.tu import TUDataset
from tqdm import tqdm

from src.data.components.p53_dataset import P53Dataset


def random_sampling(graph, nodes, walk_length, restart_prob=0.5):
    return dgl.sampling.random_walk(
        graph,
        nodes,
        length=walk_length,
        restart_prob=restart_prob
    )


def node2vec_sampling(graph, nodes, p, q, walk_length):
    return dgl.sampling.node2vec_random_walk(
        graph,
        nodes,
        p=p,
        q=q,
        walk_length=walk_length
    )


class AblationDataset(DGLDataset):
    """消融实验数据集 - 支持三种模式：
    1. 'feature_only': 使用传统游走方法（node2vec/random），但保持14维特征构建方式
    2. 'sequence_only': 直接使用游走序列作为特征
    3. 'combined': 同时使用14维特征和原始游走序列的拼接
    """

    def __init__(self,
                 name="AIDS",
                 ablation_mode='feature_only',  # 'feature_only', 'sequence_only', or 'combined'
                 walk_method='node2vec',  # 'node2vec' or 'random_walk'
                 down_sample_label=0,
                 down_sample_rate=0.1,
                 re_gen_ds_labels=False,
                 num_walks_per_node=2,
                 walk_length=10,
                 # node2vec参数
                 p=1.0,
                 q=1.0,
                 # sequence_only模式的参数
                 # sequence_dim=64,  # 序列编码的维度
                 # max_vocab_size=5000,  # 节点词汇表大小
                 # 其他参数
                 url=None,
                 raw_dir="data/raw",
                 save_dir="data/processed",
                 stage: str = "train",
                 **kwargs):
        super().__init__(name=name, url=url, raw_dir=raw_dir, save_dir=save_dir)

        self.ablation_mode = ablation_mode
        self.walk_method = walk_method
        self.num_walks_per_node = num_walks_per_node
        self.walk_length = walk_length
        self.p = p
        self.q = q
        # self.sequence_dim = sequence_dim
        # self.max_vocab_size = max_vocab_size
        self.stage = stage

        # 加载原始数据集
        if 'tox' in name.lower():
            tu_dataset_delegate = P53Dataset(name, raw_dir)
            tu_dataset_delegate.process()
        else:
            tu_dataset_delegate = TUDataset(name, raw_dir=raw_dir)

        graphs = tu_dataset_delegate.graph_lists
        labels = tu_dataset_delegate.graph_labels

        # 数据下采样
        self.cfg_seed = kwargs['seed']
        random.seed(12345)
        print('Seed fixed to 12345 for data generation')

        use_ids = []
        for i in range(len(labels)):
            if labels[i].tolist()[0] == down_sample_label:
                if random.random() <= down_sample_rate:
                    use_ids.append(i)
            else:
                use_ids.append(i)

        # 如果是sequence_only模式，需要构建全局词汇表
        # if self.ablation_mode == 'sequence_only':
        #     self.vocab = self._build_vocabulary([graphs[idx] for idx in use_ids])

        # 构建数据集
        self.graphs = []
        self.labels = []
        num_node_labels = 0
        # use_random_feat = kwargs.get('random_str_feat', False)

        print(f'Building ablation dataset with mode: {ablation_mode}, walk method: {walk_method}')

        for idx in tqdm(use_ids, desc="Processing graphs", unit="graph"):
            graph = graphs[idx]

            if 'node_labels' in graph.ndata:
                num_node_labels = max(graph.ndata['node_labels'].max(), num_node_labels)

            if self.ablation_mode == 'feature_only':
                # 模式1：使用传统游走但保持14维特征构建
                graph.ndata['sub_attr'] = self._extract_features_with_traditional_walk(graph)
            elif self.ablation_mode == 'sequence_only':
                # 模式2：直接使用游走序列作为特征
                graph.ndata['sub_attr'] = self._extract_sequence_features(graph)
            else:  # combined
                # 模式3：同时使用特征和原始序列
                graph.ndata['sub_attr'] = self._extract_combined_features(graph)

            self.graphs.append(graph.add_self_loop())
            self.labels.append(labels[idx])

        # 处理node_labels
        if 'tox' not in name.lower() and 'node_labels' in self.graphs[0].ndata:
            num_node_labels += 1
            for idx in range(len(self.graphs)):
                self.graphs[idx].ndata['node_labels'] = F.one_hot(
                    self.graphs[idx].ndata['node_labels'].squeeze(),
                    num_classes=num_node_labels
                )

        random.seed(self.cfg_seed)
        print(f'Seed reset to config value: {self.cfg_seed}')
        print(f'Dataset built with {len(self.graphs)} graphs')
        print(f'Feature dimension per node: {self.graphs[0].ndata["sub_attr"].shape[1]}')

    def _traditional_walk(self, graph: dgl.DGLGraph, nodes) -> torch.Tensor:
        if self.walk_method == 'random_walk':
            traces, _ = random_sampling(
                graph,
                nodes,
                walk_length=self.walk_length,
                restart_prob=0.5
            )
        elif self.walk_method == 'node2vec':
            traces = node2vec_sampling(
                graph,
                nodes,
                p=self.p,
                q=self.q,
                walk_length=self.walk_length,
            )
        else:
            raise NotImplementedError('Not supported walk method ' + self.walk_method)
        return traces

    def _extract_features_with_traditional_walk(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """模式1：使用传统游走但保持14维特征构建方式"""
        # all_node_features = []

        # node_sequences = []
        feature_list = []
        for _ in range(self.num_walks_per_node):
            traces = self._traditional_walk(graph, graph.nodes())
            nodes_feature_vectors = self._extract_14d_features(traces, graph)
            feature_list.append(nodes_feature_vectors)
        # node_sequence_vector = torch.cat(node_sequences, dim=0)
        # features = self._extract_14d_features(node_sequence_vector, graph)
        feature = torch.cat(feature_list, dim=1)
        return feature

    def _extract_14d_features(self, sequences: torch.Tensor, graph: dgl.DGLGraph) -> torch.Tensor:
        """从序列中提取14维特征（保持与原方法一致）"""
        features_for_all_nodes = []
        for sequence in sequences:
            features = []
            # 1. 因为没有异常评分，置为0
            # unique_transitions = len(set(zip(sequence[:-1].tolist(), sequence[1:].tolist())))
            # rarity_score = unique_transitions / max(1, len(sequence) - 1)
            features.append(0)

            # 2. 序列长度
            features.append(float(len(sequence)))

            # 3. 唯一节点数
            unique_nodes = torch.unique(sequence)
            features.append(float(len(unique_nodes)))

            # 4. 重访节点数
            features.append(float(len(sequence) - len(unique_nodes)))

            # 5. 多样性比例
            features.append(len(unique_nodes) / len(sequence))

            # 6-9. 节点度特征
            degrees = graph.in_degrees(sequence).float()
            features.extend([
                degrees.mean().item(),
                degrees.std().item() if len(degrees) > 1 else 0.0,
                degrees.max().item(),
                degrees.min().item()
            ])

            # 10-12. 序列模式特征
            pattern_features = self._extract_pattern_features(sequence)
            features.extend(pattern_features)

            # 13-14. 位置编码特征
            start_degree = graph.in_degrees(sequence[0]).item()
            end_degree = graph.in_degrees(sequence[-1]).item()
            features.extend([float(start_degree), float(end_degree)])
            features_for_all_nodes.append(torch.tensor(features, dtype=torch.float32))

        return torch.stack(features_for_all_nodes, dim=0)

    def _extract_pattern_features(self, sequence: torch.Tensor) -> List[float]:
        """提取序列模式特征"""
        if len(sequence) < 2:
            return [0.0, 0.0, 0.0]

        # 转移多样性
        transitions = [(sequence[i].item(), sequence[i + 1].item())
                       for i in range(len(sequence) - 1)]
        unique_transitions = len(set(transitions))
        transition_diversity = unique_transitions / max(1, len(transitions))

        # 回环比例
        loops = sum(1 for i in range(len(sequence) - 1)
                    if sequence[i] == sequence[i + 1])
        loop_ratio = loops / max(1, len(transitions))

        # 长距离重访
        revisits = 0
        for i, node in enumerate(sequence):
            for j in range(i + 2, len(sequence)):
                if sequence[j] == node:
                    revisits += 1
                    break
        revisit_ratio = revisits / len(sequence)

        return [transition_diversity, loop_ratio, revisit_ratio]

    def _extract_sequence_features(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """模式2：直接使用游走序列作为特征"""
        # all_node_features = []
        all_sequences = []
        for _ in range(self.num_walks_per_node):
            traces = self._traditional_walk(graph, graph.nodes())
            all_sequences.append(traces)

        all_sequences = torch.cat(all_sequences, dim=1).float()
        return all_sequences
        # for node_idx in range(graph.num_nodes()):
        #     # 执行多次游走并拼接
        #     all_sequences = []
        #
        #     for _ in range(self.num_walks_per_node):
        #         sequence = self._traditional_walk(graph, node_idx)
        #         all_sequences.extend(sequence.tolist())
        #
        #     # 将序列编码为固定维度的特征
        #     # feature = self._encode_sequence(all_sequences)
        #     all_node_features.append(feature)
        #
        # return torch.stack(all_node_features)

    def _extract_combined_features(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """模式3：同时使用14维特征和原始游走序列的拼接"""
        # 获取14维特征
        features = self._extract_features_with_traditional_walk(graph)
        
        # 获取原始序列
        sequences = self._extract_sequence_features(graph)
        
        # 按要求的顺序拼接: [feat1, feat2, ..., featN, seq1, seq2, ..., seqN]
        combined_features = torch.cat([features, sequences], dim=1)
        
        return combined_features

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx].squeeze(0)

    def __len__(self):
        return len(self.graphs)
