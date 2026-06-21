# %%
import os
import random
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
import math

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.data.tu import TUDataset
import networkx as nx
from tqdm import tqdm

from src.data.components.brains_dataset import BrainsDataset
from src.data.components.p53_dataset import P53Dataset


def _get_train_positions(num_items, frac_list=(0.7, 0.2, 0.1), shuffle=False, seed=12345):
    indices = list(range(num_items))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    train_frac = frac_list[0] if frac_list else 0.7
    train_len = int(num_items * train_frac)
    return indices[:train_len]


class AnomalyScorer:
    """计算子结构异常评分的核心类"""

    def __init__(self, graphs: List[dgl.DGLGraph], use_score='all'):
        self.graphs = graphs
        self.subgraph_freq = defaultdict(int)
        self.node_degree_dist = []
        self.global_clustering = []
        self._precompute_statistics()
        self.use_score = use_score
        if use_score == 'all':
            score_components =  '[ALL]'
        elif use_score == 'c':
            score_components = 'Complexity only'
        elif use_score == 'i':
            score_components = 'Node importance only'
        elif use_score == 'r':
            score_components = 'Rarity only'
        else:
            raise NotImplementedError
        print(f'Will use {score_components} as the anomaly score.')

    def _precompute_statistics(self):
        """预计算全局统计信息"""
        print("Precomputing global statistics...")

        for graph in self.graphs:
            # 转换为networkx以便计算复杂指标
            nx_g = dgl.to_networkx(graph.cpu())

            # 处理multigraph问题
            if nx_g.is_multigraph():
                nx_g = nx.Graph(nx_g)

            # 度分布统计
            degrees = [d for n, d in nx_g.degree()]
            self.node_degree_dist.extend(degrees)

            # 聚类系数统计
            if len(nx_g) > 0:
                try:
                    clustering = nx.average_clustering(nx_g)
                    self.global_clustering.append(clustering)
                except (nx.NetworkXError, nx.NetworkXNotImplemented) as e:
                    print(f"Warning: Cannot compute clustering for graph, skipping. Error: {e}")
                    pass

        # 计算度分布的统计量
        self.mean_degree = np.mean(self.node_degree_dist) if self.node_degree_dist else 0
        self.std_degree = np.std(self.node_degree_dist) if self.node_degree_dist else 1
        self.mean_clustering = np.mean(self.global_clustering) if self.global_clustering else 0

    def compute_sequence_anomaly_score(
            self,
            walk_sequence: torch.Tensor,
            graph: dgl.DGLGraph,
            update_frequency: bool = True
    ) -> float:
        """计算游走序列的异常评分"""
        if walk_sequence.numel() == 0:
            return 0.0
        if self.use_score == 'r' or self.use_score == 'all':
            # 1. 序列稀有度评分
            sequence_signature = self._compute_sequence_signature(walk_sequence, graph)
            freq = self.subgraph_freq.get(sequence_signature, 0)
            if update_frequency:
                freq += 1
                self.subgraph_freq[sequence_signature] = freq
            rarity_score = 1.0 / (1.0 + freq)
            composite_score = rarity_score

        if self.use_score == 'c' or self.use_score == 'all':
            # 2. 序列复杂度评分（基于转移模式）
            complexity_score = self._compute_transition_complexity(walk_sequence)
            composite_score = complexity_score

        if self.use_score == 'i' or self.use_score == 'all':
            # 3. 节点重要性评分
            importance_score = self._compute_node_importance(walk_sequence, graph)
            composite_score = importance_score

        if self.use_score == 'all':
            # 综合评分
            composite_score = 0.4 * rarity_score + 0.3 * complexity_score + 0.3 * importance_score

        return composite_score

    def _compute_sequence_signature(self, sequence: torch.Tensor, graph: dgl.DGLGraph) -> Tuple[Tuple[int, int], ...]:
        """Encode a walk by structural signals rather than graph-local node IDs."""
        if sequence.numel() == 0:
            return tuple()

        all_degrees = graph.in_degrees().float()
        max_degree = max(float(all_degrees.max().item()), 1.0) if graph.num_nodes() > 0 else 1.0
        node_ids = [int(node) for node in sequence.tolist()]
        degree_bins = torch.clamp(
            torch.floor(graph.in_degrees(node_ids).float() / max_degree * 10),
            min=0,
            max=10,
        ).int().tolist()

        seen_positions = {}
        signature = []
        for pos, node in enumerate(node_ids):
            revisit_gap = 0
            if node in seen_positions:
                revisit_gap = min(pos - seen_positions[node], 10)
            seen_positions[node] = pos
            signature.append((int(degree_bins[pos]), revisit_gap))

        return tuple(signature)

    def _compute_transition_complexity(self, sequence: torch.Tensor) -> float:
        """计算序列转移复杂度"""
        if len(sequence) < 2:
            return 0.0

        # 计算转移模式的多样性
        transitions = []
        for i in range(len(sequence) - 1):
            transitions.append((sequence[i].item(), sequence[i + 1].item()))

        # 计算唯一转移的比例
        unique_transitions = len(set(transitions))
        total_transitions = len(transitions)

        complexity = unique_transitions / max(1, total_transitions)
        return complexity

    def _compute_node_importance(self, sequence: torch.Tensor, graph: dgl.DGLGraph) -> float:
        """计算序列中节点的重要性"""
        if sequence.numel() == 0:
            return 0.0

        # 基于度的重要性
        degrees = graph.in_degrees(sequence).float()
        max_degree = graph.in_degrees().max().float()

        if max_degree > 0:
            normalized_importance = (degrees / max_degree).mean().item()
        else:
            normalized_importance = 0.0

        return normalized_importance


class SequenceBasedSampler:
    """基于序列的异常感知采样器"""

    def __init__(self, anomaly_scorer: AnomalyScorer):
        self.anomaly_scorer = anomaly_scorer

    def anomaly_guided_walk(self, graph: dgl.DGLGraph, start_node: int,
                            walk_length: Tuple[int, list], bias_strength: float = 2.0,
                            update_frequency: bool = True) -> Tuple[torch.Tensor, float]:
        """异常引导的随机游走，返回序列和异常评分"""
        current_path = [start_node]
        current_node = start_node

        # print(walk_length, type(walk_length))
        if type(walk_length) != int:
            walk_length = walk_length[0]

        for step in range(walk_length - 1):
            neighbors = graph.successors(current_node).tolist()
            if not neighbors:
                current_path.append(current_node)
                continue

            # 计算每个邻居的"潜在异常贡献"
            neighbor_scores = []
            for neighbor in neighbors:
                # 构建临时序列来评估这个选择的异常性
                temp_sequence = torch.tensor(current_path + [neighbor])
                score = self.anomaly_scorer.compute_sequence_anomaly_score(
                    temp_sequence, graph, update_frequency=update_frequency
                )
                neighbor_scores.append(score)

            # 基于异常评分的概率采样
            if max(neighbor_scores) > 0:
                scores_tensor = torch.tensor(neighbor_scores)
                probs = F.softmax(scores_tensor * bias_strength, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
            else:
                next_idx = random.randint(0, len(neighbors) - 1)

            current_node = neighbors[next_idx]
            current_path.append(current_node)

        # 计算最终序列的异常评分
        final_sequence = torch.tensor(current_path)
        final_score = self.anomaly_scorer.compute_sequence_anomaly_score(
            final_sequence, graph, update_frequency=update_frequency
        )

        return final_sequence, final_score


class SequenceFeatureExtractor:
    """序列特征提取器 - 直接处理游走序列而非统计数据"""

    def __init__(self, max_sequence_length: int = 64):
        self.max_seq_len = max_sequence_length

    def extract_sequence_features(self, sequences: List[torch.Tensor],
                                  anomaly_scores: List[float],
                                  graph: dgl.DGLGraph) -> torch.Tensor:
        """从游走序列中提取特征"""
        batch_features = []

        for seq, anomaly_score in zip(sequences, anomaly_scores):
            features = []

            # 1. 异常评分
            features.append(anomaly_score)

            # 2. 序列长度特征
            seq_len = len(seq)
            features.append(seq_len)

            # 3. 序列统计特征
            unique_nodes = torch.unique(seq)
            features.append(len(unique_nodes))  # 唯一节点数
            features.append(seq_len - len(unique_nodes))  # 重访节点数
            features.append(len(unique_nodes) / seq_len)  # 多样性比例

            # 4. 节点度特征
            degrees = graph.in_degrees(seq).float()
            features.extend([
                degrees.mean().item(),  # 平均度
                degrees.std().item(),  # 度标准差
                degrees.max().item(),  # 最大度
                degrees.min().item()  # 最小度
            ])

            # 5. 序列模式特征
            pattern_features = self._extract_pattern_features(seq)
            features.extend(pattern_features)

            # 6. 位置编码特征（序列的位置信息）
            position_features = self._extract_position_features(seq, graph)
            features.extend(position_features)

            batch_features.append(features)

        # 转换为tensor并处理NaN
        feature_tensor = torch.tensor(batch_features, dtype=torch.float32)
        feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0)

        return feature_tensor

    def _extract_pattern_features(self, sequence: torch.Tensor) -> List[float]:
        """提取序列模式特征"""
        if len(sequence) < 2:
            return [0.0, 0.0, 0.0]

        # 转移模式分析
        transitions = []
        for i in range(len(sequence) - 1):
            transitions.append((sequence[i].item(), sequence[i + 1].item()))

        # 特征计算
        unique_transitions = len(set(transitions))
        total_transitions = len(transitions)
        transition_diversity = unique_transitions / max(1, total_transitions)

        # 回环检测
        loops = sum(1 for i in range(len(sequence) - 1) if sequence[i] == sequence[i + 1])
        loop_ratio = loops / max(1, total_transitions)

        # 长距离重访
        revisits = 0
        for i, node in enumerate(sequence):
            for j in range(i + 2, len(sequence)):  # 至少间隔2步的重访
                if sequence[j] == node:
                    revisits += 1
                    break
        revisit_ratio = revisits / len(sequence)

        return [transition_diversity, loop_ratio, revisit_ratio]

    def _extract_position_features(self, sequence: torch.Tensor, graph: dgl.DGLGraph) -> List[float]:
        """提取位置编码特征"""
        if len(sequence) == 0:
            return [0.0, 0.0]

        # 起始和结束节点的重要性
        start_degree = graph.in_degrees(sequence[0]).item()
        end_degree = graph.in_degrees(sequence[-1]).item()

        return [float(start_degree), float(end_degree)]

    def pad_and_encode_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """将序列填充并编码为固定长度（可选功能，用于需要序列输入的模型）"""
        padded_sequences = []

        for seq in sequences:
            if len(seq) > self.max_seq_len:
                # 截断
                padded_seq = seq[:self.max_seq_len]
            else:
                # 填充
                padding = torch.full((self.max_seq_len - len(seq),), -1)  # 用-1填充
                padded_seq = torch.cat([seq, padding])

            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)


class SequenceBasedAnomalyAwareDataset(DGLDataset):
    """基于序列的异常感知数据集 - 直接利用游走序列而非统计数据"""

    def __init__(self, name="AIDS",
                 down_sample_label=0,
                 down_sample_rate=0.1,
                 re_gen_ds_labels=False,
                 sampling_method='sequence_anomaly_aware',
                 num_walks_per_node=2,  # 减少游走次数，每个节点只进行2次游走
                 walk_length=10,  # 使用单一游走长度
                 bias_strength=2.0,
                 max_sequence_length=64,
                 url=None,
                 raw_dir="data/raw",
                 save_dir="data/processed",
                 stage: str = "train",
                 use_score='all',
                 **kwargs):
        super().__init__(name=name, url=url, raw_dir=raw_dir, save_dir=save_dir)

        self.stage = stage
        self.sampling_method = sampling_method
        self.num_walks_per_node = num_walks_per_node
        self.walk_length = walk_length
        self.bias_strength = bias_strength
        self.use_score = use_score

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

        train_positions = _get_train_positions(
            len(use_ids),
            frac_list=kwargs.get('train_val_test_split', (0.7, 0.2, 0.1)),
            shuffle=kwargs.get('split_shuffle', False),
            seed=kwargs.get('split_seed', 12345),
        )
        train_position_set = set(train_positions)
        selected_graphs = [graphs[use_ids[pos]] for pos in train_positions]
        if not selected_graphs:
            selected_graphs = [graphs[idx] for idx in use_ids]

        # 初始化序列处理组件
        print("Initializing sequence-based anomaly-aware components...")
        self.anomaly_scorer = AnomalyScorer(selected_graphs, use_score=self.use_score)
        self.sequence_sampler = SequenceBasedSampler(self.anomaly_scorer)
        self.feature_extractor = SequenceFeatureExtractor(max_sequence_length)

        # 构建数据集
        self.graphs = [None] * len(use_ids)
        self.labels = [None] * len(use_ids)
        num_node_labels = 0
        use_random_feat = kwargs.get('random_str_feat', False)

        print(f'Building sequence-based features with {num_walks_per_node} walks per node...')

        processing_order = train_positions + [pos for pos in range(len(use_ids)) if pos not in train_position_set]

        for pos in processing_order:
            idx = use_ids[pos]
            graph = graphs[idx]
            update_frequency = pos in train_position_set

            if 'node_labels' in graph.ndata:
                num_node_labels = max(graph.ndata['node_labels'].max(), num_node_labels)

            if use_random_feat:
                # 随机特征模式
                feat_dim = num_walks_per_node * 14  # 每次游走产生14维特征
                graph.ndata['sub_attr'] = torch.randn((graph.num_nodes(), feat_dim))
            else:
                # 序列异常感知特征提取
                all_node_features = []

                for node_idx in range(graph.num_nodes()):
                    # 为每个节点进行少量高质量的异常感知游走
                    node_sequences = []
                    node_scores = []

                    for walk_idx in range(num_walks_per_node):
                        sequence, score = self.sequence_sampler.anomaly_guided_walk(
                            graph, node_idx, walk_length, bias_strength,
                            update_frequency=update_frequency
                        )
                        node_sequences.append(sequence)
                        node_scores.append(score)

                    # 提取序列特征
                    sequence_features = self.feature_extractor.extract_sequence_features(
                        node_sequences, node_scores, graph
                    )

                    # 展平所有游走的特征
                    node_feature_vector = sequence_features.flatten()
                    all_node_features.append(node_feature_vector)

                # 确保所有节点特征维度一致
                max_feat_dim = max(feat.shape[0] for feat in all_node_features)
                padded_features = []
                for feat in all_node_features:
                    if feat.shape[0] < max_feat_dim:
                        padding = torch.zeros(max_feat_dim - feat.shape[0])
                        feat = torch.cat([feat, padding])
                    padded_features.append(feat)

                graph.ndata['sub_attr'] = torch.stack(padded_features)

            self.graphs[pos] = graph.add_self_loop()
            self.labels[pos] = labels[idx]

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

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx].squeeze(0)

    def __len__(self):
        return len(self.graphs)


# 保持向后兼容性
# def random_sampling(graph, nodes, walk_length, restart_prob=0.5):
#     return dgl.sampling.random_walk(
#         graph, nodes, length=walk_length, restart_prob=restart_prob
#     )
#
#
# def node2vec_sampling(graph, nodes, p, q, walk_length):
#     return dgl.sampling.node2vec_random_walk(
#         graph, nodes, p=p, q=q, walk_length=walk_length
#     )


# 旧版本兼容 - 可以通过参数选择使用哪种实现
class AnomalyAwareGDRCDataset(DGLDataset):
    """兼容性包装器，可以选择使用序列版本或统计版本"""

    def __new__(cls, sequence_based=True, **kwargs):
        if sequence_based and kwargs.get('sampling_method') == 'anomaly_aware':
            # 使用新的序列版本
            kwargs['sampling_method'] = 'sequence_anomaly_aware'
            return SequenceBasedAnomalyAwareDataset(**kwargs)
        else:
            # 使用原来的实现（这里应该导入你原来的实现）
            # 为了简化，这里直接返回序列版本
            return SequenceBasedAnomalyAwareDataset(**kwargs)


# 修改后的关键部分 - 在SequenceBasedAnomalyAwareDataset类中

class EnhancedSequenceBasedAnomalyAwareDataset(DGLDataset):
    """增强版本的数据集，保存节点级别的异常评分供模型使用"""

    def __init__(self, *args, **kwargs):
        # 首先调用父类的初始化，但我们需要重写部分逻辑
        # 为了避免重复处理，我们直接实现完整逻辑
        # 处理不同的模式
        mode = kwargs.get('mode', 'feature_only')   # 兼容已经创建过的数据集，默认使用feature特征工程
        # optional: `r` for rarity
        # `i` for node importance
        # `c` for complexity
        # `all` for all
        use_score = kwargs.get('use_score', 'all')
        self.mode = mode
        self.use_raw_sequence_as_feature = True if mode == 'sequence_only' else False
        self.use_combined_feature = True if mode == 'combined' else False

        if self.use_combined_feature:
            print('Using combined feature')

        # 提取必要参数
        name = kwargs.get('name', "AIDS")
        raw_dir = kwargs.get('raw_dir', "data/raw")
        save_dir = kwargs.get('save_dir', "data/processed")
        url = kwargs.get('url', None)

        # 调用DGLDataset的初始化
        super().__init__(
            name=name, url=url, raw_dir=raw_dir, save_dir=save_dir
        )

        # 设置参数
        self.stage = kwargs.get('stage', 'train')
        self.sampling_method = kwargs.get('sampling_method', 'sequence_anomaly_aware')
        self.num_walks_per_node = kwargs.get('num_walks_per_node', 2)
        self.walk_length = kwargs.get('walk_length', 10)
        self.bias_strength = kwargs.get('bias_strength', 2.0)
        self.max_sequence_length = kwargs.get('max_sequence_length', 64)

        down_sample_label = kwargs.get('down_sample_label', 0)
        down_sample_rate = kwargs.get('down_sample_rate', 0.1)

        # 加载原始数据集
        if 'tox' in name.lower():
            from src.data.components.p53_dataset import P53Dataset
            tu_dataset_delegate = P53Dataset(name, raw_dir)
            tu_dataset_delegate.process()
        else:
            from dgl.data.tu import TUDataset
            tu_dataset_delegate = TUDataset(name, raw_dir=raw_dir)

        graphs = tu_dataset_delegate.graph_lists
        labels = tu_dataset_delegate.graph_labels

        # 数据下采样
        self.cfg_seed = kwargs.get('seed', 12345)
        random.seed(12345)
        print('Seed fixed to 12345 for data generation')

        use_ids = []
        for i in range(len(labels)):
            if labels[i].tolist()[0] == down_sample_label:
                if random.random() <= down_sample_rate:
                    use_ids.append(i)
            else:
                use_ids.append(i)

        train_positions = _get_train_positions(
            len(use_ids),
            frac_list=kwargs.get('train_val_test_split', (0.7, 0.2, 0.1)),
            shuffle=kwargs.get('split_shuffle', False),
            seed=kwargs.get('split_seed', 12345),
        )
        train_position_set = set(train_positions)
        selected_graphs = [graphs[use_ids[pos]] for pos in train_positions]
        if not selected_graphs:
            selected_graphs = [graphs[idx] for idx in use_ids]

        # 初始化序列处理组件
        print("Initializing enhanced sequence-based anomaly-aware components...")
        self.anomaly_scorer = AnomalyScorer(selected_graphs, use_score=use_score)
        self.sequence_sampler = SequenceBasedSampler(self.anomaly_scorer)
        self.feature_extractor = SequenceFeatureExtractor(self.max_sequence_length)

        # 构建数据集
        self.graphs = [None] * len(use_ids)
        self.labels = [None] * len(use_ids)
        num_node_labels = 0
        use_random_feat = kwargs.get('random_str_feat', False)

        print(f'Building enhanced features with {self.num_walks_per_node} walks per node...')

        processing_order = train_positions + [pos for pos in range(len(use_ids)) if pos not in train_position_set]

        for pos in tqdm(processing_order, desc="Processing graphs", unit="graph"):
            idx = use_ids[pos]
            graph = graphs[idx]
            update_frequency = pos in train_position_set

            if 'node_labels' in graph.ndata:
                num_node_labels = max(graph.ndata['node_labels'].max(), num_node_labels)

            if use_random_feat:
                # 随机特征模式
                if self.use_combined_feature:
                    # combined模式: 需要同时生成特征向量和序列长度的随机数据
                    feat_dim = self.num_walks_per_node * 14  # 特征部分
                    seq_dim = self.num_walks_per_node * self.walk_length  # 序列部分
                    total_dim = feat_dim + seq_dim
                    graph.ndata['sub_attr'] = torch.randn((graph.num_nodes(), total_dim))
                elif self.use_raw_sequence_as_feature:
                    # sequence_only模式: 只生成序列长度的随机数据
                    seq_dim = self.num_walks_per_node * self.walk_length
                    graph.ndata['sub_attr'] = torch.randn((graph.num_nodes(), seq_dim))
                else:
                    # feature_only模式: 只生成特征维度的随机数据
                    feat_dim = self.num_walks_per_node * 14
                    graph.ndata['sub_attr'] = torch.randn((graph.num_nodes(), feat_dim))
                # 即使是随机特征，也生成随机的异常评分
                graph.ndata['anomaly_scores'] = torch.rand(graph.num_nodes())
            else:
                # 增强的特征提取
                all_node_features = []
                all_node_anomaly_scores = []  # 新增：保存每个节点的异常评分

                for node_idx in range(graph.num_nodes()):
                    node_sequences = []
                    node_scores = []

                    for walk_idx in range(self.num_walks_per_node):
                        sequence, score = self.sequence_sampler.anomaly_guided_walk(
                            graph, node_idx, self.walk_length, self.bias_strength,
                            update_frequency=update_frequency
                        )
                        node_sequences.append(sequence)
                        node_scores.append(score)

                    # 新增：计算节点的综合异常评分
                    # 使用多次游走的平均异常评分作为节点的异常评分
                    node_anomaly_score = torch.tensor(node_scores).mean()
                    all_node_anomaly_scores.append(node_anomaly_score)

                    if self.use_raw_sequence_as_feature:
                        # 使用原始的游走序列
                        cat_seq = torch.cat(node_sequences)
                        all_node_features.append(cat_seq)
                    elif self.use_combined_feature:
                        # 模式combined: 同时使用特征和原始序列
                        # 先提取特征
                        sequence_features = self.feature_extractor.extract_sequence_features(
                            node_sequences, node_scores, graph
                        )
                        feature_vector = sequence_features.flatten()
                        
                        # 再获取原始序列
                        raw_sequences = torch.cat(node_sequences)
                        
                        # 按要求的顺序拼接: [feat1, feat2, ..., featN, seq1, seq2, ..., seqN]
                        combined_feature = torch.cat([feature_vector, raw_sequences.float()])
                        all_node_features.append(combined_feature)
                    else:
                        # 提取序列特征
                        sequence_features = self.feature_extractor.extract_sequence_features(
                            node_sequences, node_scores, graph
                        )

                        # 展平所有游走的特征
                        node_feature_vector = sequence_features.flatten()
                        all_node_features.append(node_feature_vector)

                if self.use_raw_sequence_as_feature or self.use_combined_feature:
                    # 对于sequence_only模式和combined模式，直接使用堆叠的特征
                    graph.ndata['sub_attr'] = torch.stack(all_node_features).float()
                else:
                    # 确保所有节点特征维度一致 (feature_only模式)
                    max_feat_dim = max(feat.shape[0] for feat in all_node_features)
                    padded_features = []
                    for feat in all_node_features:
                        if feat.shape[0] < max_feat_dim:
                            padding = torch.zeros(max_feat_dim - feat.shape[0])
                            feat = torch.cat([feat, padding])
                        padded_features.append(feat)
                    graph.ndata['sub_attr'] = torch.stack(padded_features)
                # 新增：保存节点异常评分
                graph.ndata['anomaly_scores'] = torch.stack(all_node_anomaly_scores)

                # 计算并保存图级别的异常统计信息
                graph_anomaly_stats = self._compute_graph_anomaly_stats(graph)
                graph.graph_anomaly_stats = graph_anomaly_stats

            self.graphs[pos] = graph.add_self_loop()
            self.labels[pos] = labels[idx]

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
        print(f'Enhanced dataset built with {len(self.graphs)} graphs')
        print(f'Mode: {self.mode}')
        print(f'Feature dimension per node: {self.graphs[0].ndata["sub_attr"].shape[1]}')
        print(f'Anomaly score dimension per node: {self.graphs[0].ndata["anomaly_scores"].shape[0]}')

    def _compute_graph_anomaly_stats(self, graph: dgl.DGLGraph) -> Dict[str, float]:
        """计算图级别的异常统计信息"""
        anomaly_scores = graph.ndata['anomaly_scores']

        stats = {
            'mean_anomaly': float(anomaly_scores.mean()),
            'std_anomaly': float(anomaly_scores.std()),
            'max_anomaly': float(anomaly_scores.max()),
            'min_anomaly': float(anomaly_scores.min()),
            'high_anomaly_ratio': float((anomaly_scores > 0.5).float().mean()),
            'very_high_anomaly_ratio': float((anomaly_scores > 0.8).float().mean())
        }

        return stats

    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息，供模型初始化使用"""
        if len(self.graphs) > 0:
            sample_graph = self.graphs[0]
            info = {
                'sub_attr_dim': sample_graph.ndata['sub_attr'].shape[1],
                'has_anomaly_scores': 'anomaly_scores' in sample_graph.ndata,
                'has_graph_stats': hasattr(sample_graph, 'graph_anomaly_stats')
            }
            return info
        return {}

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx].squeeze(0)

    def __len__(self):
        return len(self.graphs)
