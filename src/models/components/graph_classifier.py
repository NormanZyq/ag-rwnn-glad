import math

import dgl
import torch
from dgl.nn.pytorch import GlobalAttentionPooling
import torch.nn.functional as F


class GraphClassifier(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            dropout=0.2,
            use_multi_readout: bool = False,
            use_anomaly_readout: bool = False,
            anomaly_topk_ratio: float = 0.2,
            anomaly_temperature: float = 2.0,
            min_topk: int = 1,
            use_readout_norm: bool = False,
    ):
        super().__init__()

        input_size = eval(input_size) if isinstance(input_size, str) else input_size
        self.input_size = input_size
        self.use_multi_readout = use_multi_readout
        self.use_anomaly_readout = use_anomaly_readout
        self.anomaly_topk_ratio = anomaly_topk_ratio
        self.anomaly_temperature = anomaly_temperature
        self.min_topk = min_topk
        self.use_readout_norm = use_readout_norm

        # 使用GlobalAttentionPooling替代AvgPooling
        # gate_nn用于计算每个节点的注意力权重
        gate_nn = torch.nn.Linear(input_size, 1)
        # self.avg_pool = AvgPooling()        # avg pooling
        self.attention_pool = GlobalAttentionPooling(gate_nn)   # attention-based pooling

        readout_multiplier = 1
        if self.use_multi_readout:
            readout_multiplier += 3  # mean, max, std
        if self.use_anomaly_readout:
            readout_multiplier += 2  # anomaly-weighted mean, anomaly top-k mean

        classifier_input_size = input_size * readout_multiplier
        self.readout_norm = torch.nn.LayerNorm(classifier_input_size) if self.use_readout_norm else torch.nn.Identity()
        self.linear = torch.nn.Linear(classifier_input_size, output_size)
        self.dropout = torch.nn.Dropout(p=dropout)

    @staticmethod
    def _sanitize_tensor(tensor: torch.Tensor, nan: float = 0.0, posinf: float = 0.0,
                         neginf: float = 0.0) -> torch.Tensor:
        return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)

    @staticmethod
    def _sanitize_logits(tensor: torch.Tensor, logit_clip: float = 30.0) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=logit_clip, neginf=-logit_clip)
        return torch.clamp(tensor, min=-logit_clip, max=logit_clip)

    def _basic_readouts(self, g: dgl.DGLGraph, feat: torch.Tensor):
        feat = self._sanitize_tensor(feat)
        with g.local_scope():
            g.ndata['_readout_feat'] = feat
            mean_pool = dgl.readout_nodes(g, '_readout_feat', op='mean')
            max_pool = dgl.readout_nodes(g, '_readout_feat', op='max')

            g.ndata['_readout_feat_sq'] = feat * feat
            mean_sq_pool = dgl.readout_nodes(g, '_readout_feat_sq', op='mean')
            std_pool = torch.sqrt(torch.clamp(mean_sq_pool - mean_pool.pow(2), min=0.0))

        return (
            self._sanitize_tensor(mean_pool),
            self._sanitize_tensor(max_pool),
            self._sanitize_tensor(std_pool),
        )

    def _anomaly_readouts(
            self,
            g: dgl.DGLGraph,
            feat: torch.Tensor,
            anomaly_scores: torch.Tensor,
    ):
        node_counts = g.batch_num_nodes().tolist()
        if anomaly_scores is None:
            batch_size = len(node_counts)
            zero_pool = feat.new_zeros((batch_size, self.input_size))
            return zero_pool, zero_pool

        anomaly_scores = self._sanitize_tensor(
            anomaly_scores.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        if anomaly_scores.numel() != feat.shape[0]:
            raise ValueError(
                f"Expected {feat.shape[0]} anomaly scores, got {anomaly_scores.numel()}."
            )

        feat = self._sanitize_tensor(feat)
        feat_chunks = torch.split(feat, node_counts, dim=0)
        score_chunks = torch.split(anomaly_scores, node_counts, dim=0)

        weighted_pools = []
        topk_pools = []
        for node_feat, node_score in zip(feat_chunks, score_chunks):
            if node_feat.numel() == 0:
                weighted_pools.append(feat.new_zeros(self.input_size))
                topk_pools.append(feat.new_zeros(self.input_size))
                continue

            weights = torch.softmax(node_score * self.anomaly_temperature, dim=0).unsqueeze(-1)
            weighted_pools.append(torch.sum(node_feat * weights, dim=0))

            topk = max(self.min_topk, math.ceil(node_feat.shape[0] * self.anomaly_topk_ratio))
            topk = min(topk, node_feat.shape[0])
            topk_idx = torch.topk(node_score, k=topk, largest=True).indices
            topk_pools.append(node_feat[topk_idx].mean(dim=0))

        return (
            self._sanitize_tensor(torch.stack(weighted_pools, dim=0)),
            self._sanitize_tensor(torch.stack(topk_pools, dim=0)),
        )

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor, anomaly_scores: torch.Tensor = None):
        if feat.dim() > 2:
            feat = feat.squeeze()
        feat = self._sanitize_tensor(feat)

        readouts = []
        readouts.append(self.attention_pool(g, feat))      # attention-based pooling

        if self.use_multi_readout:
            readouts.extend(self._basic_readouts(g, feat))

        if self.use_anomaly_readout:
            readouts.extend(self._anomaly_readouts(g, feat, anomaly_scores))

        out = torch.cat(readouts, dim=1)
        out = self._sanitize_tensor(out)
        out = self.readout_norm(out)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.linear(out)
        return self._sanitize_logits(out)
