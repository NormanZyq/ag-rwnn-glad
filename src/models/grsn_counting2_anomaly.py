import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import lightning.pytorch as pl
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import AUROC, F1Score, Accuracy, Precision, Recall
from typing import Optional, Sequence, Tuple, Dict, Any


class AnomalyAwareGRSNCounting(pl.LightningModule):
    """异常感知的图级异常检测模型，结合数据增强的异常信息"""

    def __init__(
            self,
            attr_feat_size: int,
            structure_feat_size: int,
            structure_hidden_size,
            str_dropout: float,
            graph_conv: torch.nn.Module,
            graph_classifier: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            pos_weight=1.0,
            class_weights: Optional[Sequence[float]] = None,
            # 新增参数
            use_anomaly_aware_recon: bool = False,
            use_anomaly_scores: bool = True,
            use_score_components: bool = False,
            use_learnable_score_fusion: bool = False,
            score_component_size: int = 3,
            anomaly_threshold: float = 0.5,
            adaptive_lambda_range: Tuple[float, float] = (0.01, 1.0)
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.graph_conv = graph_conv
        self.graph_classifier = graph_classifier

        # hard code
        self.label_w = 1
        self.str_w = 0

        if isinstance(structure_feat_size, str):
            self.structure_feat_size = structure_feat_size = eval(structure_feat_size)
        else:
            self.structure_feat_size = structure_feat_size

        assert (attr_feat_size > 0) or (structure_feat_size > 0)
        self.attr_feat_size = attr_feat_size

        self.use_anomaly_scores = use_anomaly_scores
        self.use_score_components = use_score_components
        self.use_learnable_score_fusion = use_learnable_score_fusion
        self.score_component_size = int(score_component_size)

        if self.use_learnable_score_fusion:
            self.score_component_logits = nn.Parameter(torch.zeros(self.score_component_size))

        # 原始网络结构
        self.W1 = torch.nn.Sequential(
            torch.nn.Linear(structure_feat_size, structure_hidden_size),
            torch.nn.Dropout(p=str_dropout),
            torch.nn.LeakyReLU()
        )

        base_fusion_size = attr_feat_size + structure_hidden_size
        fusion_input_size = base_fusion_size
        if self.use_anomaly_scores:
            fusion_input_size += 1
        if self.use_score_components:
            fusion_input_size += self.score_component_size

        self.fusion_layer = torch.nn.Linear(
            fusion_input_size,
            base_fusion_size
        )

        # 异常感知组件
        self.use_anomaly_aware_recon = use_anomaly_aware_recon
        self.anomaly_threshold = anomaly_threshold
        self.adaptive_lambda_range = adaptive_lambda_range

        if use_anomaly_aware_recon:
            # 异常评分提取器 - 从节点特征中提取异常评分
            self.anomaly_score_extractor = nn.Sequential(
                nn.Linear(structure_feat_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        #
        #     # 自适应lambda网络 - 根据图的异常特征动态调整重构权重
        #     self.adaptive_lambda_net = nn.Sequential(
        #         nn.Linear(4, 32),  # 输入：图级异常统计
        #         nn.ReLU(),
        #         nn.Linear(32, 16),
        #         nn.ReLU(),
        #         nn.Linear(16, 1),
        #         nn.Sigmoid()
        #     )
        #
            # 异常感知的图编码器 - 考虑节点异常性的图表示学习
            self.anomaly_aware_encoder = nn.Sequential(
                nn.Linear(attr_feat_size + structure_hidden_size + 1, 128),  # +1 for anomaly score
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, attr_feat_size + structure_hidden_size)
            )
        # else:
        #     self._lambda = torch.nn.Parameter(torch.tensor(0.1).clamp(0, 1))

        # 损失函数 - 支持类别不平衡；pos_weight 兼容旧配置，表示 class-1 的权重。
        class_weight_tensor = self._build_class_weight(class_weights, pos_weight)
        self.label_crit = nn.CrossEntropyLoss(weight=class_weight_tensor)
        self.struct_crit = torch.nn.MSELoss()

        # 评估指标
        self._init_metrics()

    @staticmethod
    def _sanitize_tensor(tensor: torch.Tensor, nan: float = 0.0, posinf: float = 0.0,
                         neginf: float = 0.0) -> torch.Tensor:
        return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)

    @staticmethod
    def _sanitize_logits(tensor: torch.Tensor, logit_clip: float = 30.0) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=logit_clip, neginf=-logit_clip)
        return torch.clamp(tensor, min=-logit_clip, max=logit_clip)

    @staticmethod
    def _prediction_probabilities(labels_pred: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(labels_pred, dim=1)
        return torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    @staticmethod
    def _build_class_weight(class_weights, pos_weight):
        if class_weights is not None:
            weights = torch.as_tensor(list(class_weights), dtype=torch.float32)
            if weights.numel() != 2:
                raise ValueError("class_weights must contain exactly two values for binary graph labels.")
            return weights

        if pos_weight is None:
            return None

        return torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)

    def _init_metrics(self):
        """初始化评估指标"""
        # Loss metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # AUC - 改为多类别
        self.train_auc = AUROC(task='multiclass', num_classes=2)
        self.val_auc = AUROC(task='multiclass', num_classes=2)
        self.test_auc = AUROC(task='multiclass', num_classes=2)

        # Other metrics - 改为多类别
        for stage in ['train', 'val', 'test']:
            setattr(self, f'{stage}_precision', Precision(task='multiclass', num_classes=2))
            setattr(self, f'{stage}_recall', Recall(task='multiclass', num_classes=2))
            setattr(self, f'{stage}_acc', Accuracy(task='multiclass', num_classes=2))
            setattr(self, f'{stage}_f1', F1Score(task='multiclass', num_classes=2))

        self.val_auc_best = MaxMetric()

    def extract_score_components(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """获取节点级异常评分组成项：rarity、complexity、importance。"""
        if 'anomaly_score_components' in graph.ndata:
            components = graph.ndata['anomaly_score_components'].float()
            if components.dim() == 1:
                components = components.unsqueeze(-1)
            if components.shape[1] != self.score_component_size:
                raise ValueError(
                    f"Expected anomaly_score_components with {self.score_component_size} columns, "
                    f"got {components.shape[1]}."
                )
            return self._sanitize_tensor(components, nan=0.0, posinf=1.0, neginf=0.0)

        return torch.zeros(
            graph.num_nodes(),
            self.score_component_size,
            device=graph.device,
            dtype=torch.float32,
        )

    def extract_anomaly_scores(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """从节点特征中提取异常评分"""
        if self.use_learnable_score_fusion and 'anomaly_score_components' in graph.ndata:
            components = self.extract_score_components(graph)
            weights = F.softmax(self.score_component_logits, dim=0)
            anomaly_scores = torch.sum(components * weights, dim=1)
            return self._sanitize_tensor(anomaly_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if 'anomaly_scores' in graph.ndata:
            anomaly_scores = graph.ndata['anomaly_scores'].float()
            anomaly_scores = self._sanitize_tensor(anomaly_scores.view(-1), nan=0.0, posinf=1.0, neginf=0.0)
            return anomaly_scores.clamp(0.0, 1.0)

        if 'sub_attr' in graph.ndata and self.use_anomaly_aware_recon:
            # 使用神经网络从结构特征中提取异常评分
            sub_attr = self._sanitize_tensor(graph.ndata['sub_attr'].float())
            anomaly_scores = self.anomaly_score_extractor(sub_attr)
            return self._sanitize_tensor(anomaly_scores.squeeze(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            # 如果没有结构特征或不使用异常感知，返回均匀评分
            return torch.ones(graph.num_nodes(), device=graph.device) * 0.5

    def compute_graph_anomaly_stats(self, anomaly_scores: torch.Tensor) -> torch.Tensor:
        """计算图级别的异常统计信息"""
        stats = torch.tensor([
            anomaly_scores.mean().item(),  # 平均异常评分
            anomaly_scores.std(unbiased=False).item(),  # 异常评分标准差
            (anomaly_scores > self.anomaly_threshold).float().mean().item(),  # 高异常节点比例
            anomaly_scores.max().item() - anomaly_scores.min().item()  # 异常评分范围
        ], device=anomaly_scores.device)

        return stats

    def get_h_graph(self, graph: dgl.DGLGraph, anomaly_scores: torch.Tensor = None):
        """获取图表示，可选地考虑异常评分"""
        # 原始特征拼接
        feat = self.concat_attrs(graph)
        feat = self._sanitize_tensor(feat)

        if self.use_anomaly_scores and anomaly_scores is not None:
            feat = torch.cat([feat, anomaly_scores.unsqueeze(-1)], dim=-1)

        if self.use_score_components:
            score_components = self.extract_score_components(graph)
            feat = torch.cat([feat, score_components], dim=-1)

        # 融合层
        feat = self._sanitize_tensor(feat)
        feat = self.fusion_layer(feat)
        feat = F.leaky_relu(feat)
        feat = self._sanitize_tensor(feat)

        # 考虑邻接矩阵
        feat = torch.mm(graph.adjacency_matrix().to_dense(), feat)
        feat = self._sanitize_tensor(feat)

        # 图卷积
        h_graph = self.graph_conv(graph, feat)
        h_graph = self._sanitize_tensor(h_graph)

        return h_graph

    def concat_attrs(self, graph):
        """拼接属性特征"""
        feat_str = feat_node_attr = feat_node_labels = None

        if 'sub_attr' in graph.ndata and self.structure_feat_size > 0:
            feat_str = self._sanitize_tensor(graph.ndata['sub_attr'].float())
            feat_str = F.normalize(feat_str, p=2, dim=1)
            feat_str = self.W1(feat_str)
            feat_str = self._sanitize_tensor(feat_str)

        if 'node_attr' in graph.ndata and self.attr_feat_size > 0:
            feat_node_attr = self._sanitize_tensor(graph.ndata['node_attr'].float())

        if 'node_labels' in graph.ndata and self.attr_feat_size > 0:
            feat_node_labels = self._sanitize_tensor(graph.ndata['node_labels'].float())

        feats_tmp = [feat_node_attr, feat_node_labels, feat_str]
        feat = []
        for f in feats_tmp:
            if f is not None:
                feat.append(f)

        feat = torch.cat(feat, dim=1)
        return self._sanitize_tensor(feat)

    def forward(self, graph: dgl.DGLGraph, **kwargs):
        """前向传播"""
        # 提取异常评分
        anomaly_scores = self.extract_anomaly_scores(graph)

        # 获取图表示
        h_graph = self.get_h_graph(graph, anomaly_scores)

        return h_graph, anomaly_scores

    def model_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """模型步骤 - 修复损失计算"""
        graph, labels = batch
        labels = labels.long()  # 改为long类型用于CrossEntropyLoss

        # 前向传播
        h_graph, anomaly_scores = self.forward(graph)

        # 计算A_hat，计算结构损失
        if self.str_w > 0:
            A_hat = torch.mm(h_graph, h_graph.T)  # [N, N]
            A_hat = F.sigmoid(A_hat)
            loss_str = self.struct_crit(graph.adjacency_matrix().to_dense(), A_hat)
        else:
            loss_str = 0

        # 预测标签 - 输出维度改为2，不再squeeze
        labels_pred = self.graph_classifier(graph, h_graph, anomaly_scores=anomaly_scores)
        labels_pred = self._sanitize_logits(labels_pred)

        # 计算损失
        loss_label = self.label_crit(labels_pred, labels)

        # 额外的统计信息
        extra_info = {
            # 'adaptive_lambda': adaptive_lambda.item() if isinstance(adaptive_lambda, torch.Tensor) else adaptive_lambda,
            'mean_anomaly_score': anomaly_scores.mean().item(),
            'high_anomaly_ratio': (anomaly_scores > self.anomaly_threshold).float().mean().item(),
            # 'reconstruction_quality': F.mse_loss(A_hat, graph.adjacency_matrix().to_dense()).item()
        }

        return loss_label, loss_str, labels_pred, extra_info

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        _, labels = batch
        loss_label, loss_str, labels_pred, extra_info = self.model_step(batch)

        loss = self.label_w * loss_label + self.str_w * loss_str

        # 记录损失
        self.train_loss(loss.item())

        # 使用softmax获取预测概率，然后取argmax获取类别
        labels_pred_prob = self._prediction_probabilities(labels_pred)
        self.train_auc(labels_pred_prob, labels)
        labels_pred_class = torch.argmax(labels_pred_prob, dim=1)

        # 更新指标
        self.train_precision(labels_pred_class, labels)
        self.train_recall(labels_pred_class, labels)
        self.train_acc(labels_pred_class, labels)
        self.train_f1(labels_pred_class, labels)

        # 日志记录
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('train/auc', self.train_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

        # if self.use_anomaly_aware_recon:
        #     self.log('train/adaptive_lambda', extra_info['adaptive_lambda'], on_step=True, on_epoch=True)
        #     self.log('train/mean_anomaly_score', extra_info['mean_anomaly_score'], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """验证步骤"""
        _, labels = batch
        loss_label, loss_str, labels_pred, extra_info = self.model_step(batch)

        loss = self.label_w * loss_label + self.str_w * loss_str

        self.val_loss(loss.item())

        # 使用softmax获取预测概率，然后取argmax获取类别
        labels_pred_prob = self._prediction_probabilities(labels_pred)
        self.val_auc(labels_pred_prob, labels)
        labels_pred_class = torch.argmax(labels_pred_prob, dim=1)

        self.val_precision(labels_pred_class, labels)
        self.val_recall(labels_pred_class, labels)
        self.val_acc(labels_pred_class, labels)
        self.val_f1(labels_pred_class, labels)
        self.log(
            'val/prob_std',
            labels_pred_prob[:, 1].std(unbiased=False),
            on_step=False,
            on_epoch=True,
            batch_size=1,
            prog_bar=False,
        )

        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

        # if self.use_anomaly_aware_recon:
        #     self.log('val/adaptive_lambda', extra_info['adaptive_lambda'], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """测试步骤"""
        graph, labels = batch
        loss_label, loss_str, labels_pred, extra_info = self.model_step(batch)

        loss = self.label_w * loss_label + self.str_w * loss_str

        self.test_loss(loss.item())

        # 使用softmax获取预测概率，然后取argmax获取类别
        labels_pred_prob = self._prediction_probabilities(labels_pred)
        self.test_auc(labels_pred_prob, labels)
        labels_pred_class = torch.argmax(labels_pred_prob, dim=1)

        self.test_precision(labels_pred_class, labels)
        self.test_recall(labels_pred_class, labels)
        self.test_acc(labels_pred_class, labels)
        self.test_f1(labels_pred_class, labels)

        # 日志记录
        self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时的回调"""
        auc = self.val_auc.compute()
        self.val_auc_best(auc)
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器"""
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}
