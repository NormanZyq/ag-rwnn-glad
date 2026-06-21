import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader
from lightning import LightningDataModule
from torch.utils.data import Dataset

from src.data.components.ablation_dataset import AblationDataset


class AblationDataModule(LightningDataModule):
    """消融实验数据模块 - 支持三种消融实验模式"""
    
    def __init__(
        self,
        # 基础参数
        name: str = 'AIDS',
        dsl: int = 0,
        down_sample_rate: float = 0.1,
        re_gen_ds_labels=False,
        # 消融实验参数
        ablation_mode: str = 'feature_only',  # 'feature_only', 'sequence_only', or 'combined'
        walk_method: str = 'node2vec',  # 'node2vec' or 'random_walk'
        # 游走参数
        num_walks_per_node: int = 2,
        walk_length: int = 10,
        # Node2Vec参数
        p: float = 1.0,
        q: float = 1.0,
        # 数据集参数
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        shuffle: bool = False,
        seed: int = 12345,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters(logger=False)
        
        self.name = name
        self.ablation_mode = ablation_mode
        self.walk_method = walk_method
        self.down_sample_label = dsl
        self.down_sample_rate = down_sample_rate
        self.re_gen_ds_labels = re_gen_ds_labels
        self.num_walks_per_node = num_walks_per_node
        self.walk_length = walk_length
        self.p = p
        self.q = q
        # self.sequence_dim = sequence_dim
        # self.max_vocab_size = max_vocab_size
        self.data_dir = data_dir[:-1] if data_dir.endswith('/') else data_dir
        
        # 数据分割
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.kwargs = kwargs
        
    def _generate_cache_filename(self) -> str:
        """生成缓存文件名"""
        base_params = (
            f'{self.data_dir}/processed/{self.name}'
            f'-ablation={self.ablation_mode}'
            f'-walk={self.walk_method}'
        )
        
        if self.ablation_mode == 'feature_only':
            # 特征模式：包含游走参数
            if self.walk_method == 'node2vec':
                filename = (
                    f'{base_params}'
                    f'-p={self.p}-q={self.q}'
                    f'-dsl={self.down_sample_label}'
                    f'-rate={self.down_sample_rate}'
                    f'-walks={self.num_walks_per_node}'
                    f'-length={self.walk_length}.pkl'
                )
            else:  # random_walk
                filename = (
                    f'{base_params}'
                    f'-dsl={self.down_sample_label}'
                    f'-rate={self.down_sample_rate}'
                    f'-walks={self.num_walks_per_node}'
                    f'-length={self.walk_length}.pkl'
                )
        elif self.ablation_mode == 'sequence_only':
            # 序列模式：包含序列维度参数
            if self.walk_method == 'node2vec':
                filename = (
                    f'{base_params}'
                    f'-p={self.p}-q={self.q}'
                    f'-dsl={self.down_sample_label}'
                    f'-rate={self.down_sample_rate}'
                    f'-walks={self.num_walks_per_node}'
                    f'-length={self.walk_length}.pkl'
                )
            else:  # random_walk
                filename = (
                    f'{base_params}'
                    f'-dsl={self.down_sample_label}'
                    f'-rate={self.down_sample_rate}'
                    f'-walks={self.num_walks_per_node}'
                    f'-length={self.walk_length}.pkl'
                )
        else:  # combined
            # 组合模式：包含所有参数
            if self.walk_method == 'node2vec':
                filename = (
                    f'{base_params}'
                    f'-p={self.p}-q={self.q}'
                    f'-dsl={self.down_sample_label}'
                    f'-rate={self.down_sample_rate}'
                    f'-walks={self.num_walks_per_node}'
                    f'-length={self.walk_length}.pkl'
                )
            else:  # random_walk
                filename = (
                    f'{base_params}'
                    f'-dsl={self.down_sample_label}'
                    f'-rate={self.down_sample_rate}'
                    f'-walks={self.num_walks_per_node}'
                    f'-length={self.walk_length}.pkl'
                )
                
        return filename
    
    def prepare_data(self) -> None:
        """下载数据（如果需要）"""
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        """加载和分割数据集"""
        if not self.data_train and not self.data_val and not self.data_test:
            
            # 生成缓存文件名
            expect_file_name = self._generate_cache_filename()
            
            # 检查是否使用随机特征
            use_random_feat = self.kwargs.get('random_str_feat', False)
            
            # 尝试加载缓存
            if os.path.exists(expect_file_name) and not use_random_feat:
                print(f"Loading cached ablation dataset from: {expect_file_name}")
                with open(expect_file_name, 'rb') as f:
                    dataset = pickle.load(f)
            else:
                print(f"Creating new ablation dataset:")
                print(f"  Mode: {self.ablation_mode}")
                print(f"  Walk method: {self.walk_method}")
                
                # 构建数据集参数
                dataset_kwargs = {
                    'name': self.name,
                    'ablation_mode': self.ablation_mode,
                    'walk_method': self.walk_method,
                    'down_sample_label': self.down_sample_label,
                    'down_sample_rate': self.down_sample_rate,
                    're_gen_ds_labels': self.re_gen_ds_labels,
                    'num_walks_per_node': self.num_walks_per_node,
                    'walk_length': self.walk_length,
                    'p': self.p,
                    'q': self.q,
                    'seed': self.hparams.seed,
                    'raw_dir': self.data_dir + '/raw',
                    'save_dir': self.data_dir + '/processed',
                }
                
                # 添加其他kwargs
                dataset_kwargs.update(self.kwargs)
                
                # 创建数据集
                dataset = AblationDataset(**dataset_kwargs)
                
                # 保存缓存
                if not use_random_feat and self.num_walks_per_node <= 30:
                    os.makedirs(os.path.dirname(expect_file_name), exist_ok=True)
                    print(f"Saving dataset cache to: {expect_file_name}")
                    with open(expect_file_name, 'wb') as f:
                        pickle.dump(dataset, f)
            
            # 分割数据集
            self.data_train, self.data_val, self.data_test = split_dataset(
                dataset=dataset,
                frac_list=self.hparams.train_val_test_split,
                shuffle=self.hparams.shuffle,
                random_state=12345  # 固定随机种子
            )
            
        # 打印统计信息
        self._print_dataset_statistics()
    
    def _print_dataset_statistics(self):
        """打印数据集统计信息"""
        num_train_anomaly = sum(1 for s in self.data_train if s[1].numpy() == self.down_sample_label)
        num_val_anomaly = sum(1 for s in self.data_val if s[1].numpy() == self.down_sample_label)
        num_test_anomaly = sum(1 for s in self.data_test if s[1].numpy() == self.down_sample_label)
        
        print(f'''
        ============= Ablation Dataset Statistics =============
        Ablation Mode: {self.ablation_mode}
        Walk Method: {self.walk_method}
        {'Node2Vec params: p=' + str(self.p) + ', q=' + str(self.q) if self.walk_method == 'node2vec' else ''}
        
        Train: Normal={len(self.data_train) - num_train_anomaly}, Anomaly={num_train_anomaly}, 
               Total={len(self.data_train)}, Anomaly Rate={num_train_anomaly / len(self.data_train):.2%}
               
        Val:   Normal={len(self.data_val) - num_val_anomaly}, Anomaly={num_val_anomaly}, 
               Total={len(self.data_val)}, Anomaly Rate={num_val_anomaly / len(self.data_val):.2%}
               
        Test:  Normal={len(self.data_test) - num_test_anomaly}, Anomaly={num_test_anomaly}, 
               Total={len(self.data_test)}, Anomaly Rate={num_test_anomaly / len(self.data_test):.2%}
        ======================================================
        ''')
        
        # 打印特征维度信息
        if self.data_train:
            sample_graph, _ = self.data_train[0]
            if 'sub_attr' in sample_graph.ndata:
                print(f"Feature dimension: {sample_graph.ndata['sub_attr'].shape[1]}")
    
    def train_dataloader(self) -> GraphDataLoader:
        """创建训练数据加载器"""
        return GraphDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self) -> GraphDataLoader:
        """创建验证数据加载器"""
        return GraphDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self) -> GraphDataLoader:
        """创建测试数据加载器"""
        return GraphDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """清理资源"""
        pass
    
    def state_dict(self) -> Dict[Any, Any]:
        """保存状态"""
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态"""
        pass
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        if hasattr(self, 'data_train') and self.data_train is not None:
            sample_graph, _ = self.data_train[0]
            if 'sub_attr' in sample_graph.ndata:
                return sample_graph.ndata['sub_attr'].shape[1]
                
        # 默认维度估算
        if self.ablation_mode == 'feature_only':
            # 14维特征 * 游走次数
            return 14 * self.num_walks_per_node
        elif self.ablation_mode == 'sequence_only':
            # 序列长度 * 游走次数
            return self.walk_length * self.num_walks_per_node
        else:  # combined
            # 特征维度 + 序列维度
            feat_dim = 14 * self.num_walks_per_node
            seq_dim = self.walk_length * self.num_walks_per_node
            return feat_dim + seq_dim

if __name__ == '__main__':
    # ablation1_node2vec = AblationDataModule(
    #     name='AIDS',
    #     ablation_mode='feature_only',  # 保持14维特征构建
    #     walk_method='node2vec',  # 但使用node2vec游走
    #     p=2.0,
    #     q=0.6,
    #     num_walks_per_node=2,
    #     walk_length=10,
    #     batch_size=32,
    # )
    # ablation1_node2vec.setup()
    # print(ablation1_node2vec.train_dataloader())

    # ablation1_random_walk = AblationDataModule(
    #     name='AIDS',
    #     ablation_mode='feature_only',
    #     walk_method='random_walk',
    #     # p=2.0,
    #     # q=0.6,
    #     num_walks_per_node=2,
    #     walk_length=10,
    #     batch_size=32,
    # )
    # ablation1_random_walk.setup()


    ablation2_node2vec = AblationDataModule(
        name='AIDS',
        ablation_mode='sequence_only',
        walk_method='node2vec',
        p=2.0,
        q=0.6,
        num_walks_per_node=2,
        walk_length=10,
        batch_size=32,
    )
    ablation2_node2vec.setup()

    ablation2_random_walk = AblationDataModule(
        name='AIDS',
        ablation_mode='sequence_only',
        walk_method='random_walk',
        # p=2.0,
        # q=0.6,
        num_walks_per_node=2,
        walk_length=10,
        batch_size=32,
    )
    ablation2_random_walk.setup()