# %%
import os
import pickle
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader
from lightning import LightningDataModule
from torch.utils.data import Dataset

# 修改import - 导入新的AnomalyAwareGDRCDataset
from src.data.components.gdrc_multi_dataset import AnomalyAwareGDRCDataset, EnhancedSequenceBasedAnomalyAwareDataset


def _load_pickle_with_numpy_compat(path: str) -> Any:
    """Load pickle cache and fallback for NumPy module-path incompatibility."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith('numpy._core'):
            raise

    print("Detected legacy NumPy pickle module path 'numpy._core'; retrying with compatibility mapping.")
    original_core = sys.modules.get('numpy._core')
    original_multiarray = sys.modules.get('numpy._core.multiarray')

    # Map newer pickle module path to the current NumPy layout during loading.
    sys.modules['numpy._core'] = np.core
    import numpy.core.multiarray as np_multiarray
    sys.modules['numpy._core.multiarray'] = np_multiarray

    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        if original_core is None:
            sys.modules.pop('numpy._core', None)
        else:
            sys.modules['numpy._core'] = original_core

        if original_multiarray is None:
            sys.modules.pop('numpy._core.multiarray', None)
        else:
            sys.modules['numpy._core.multiarray'] = original_multiarray


class GDRCMDataModule(LightningDataModule):
    def __init__(
            self,
            name: str = 'AIDS',
            dsl: int = 0,
            down_sample_rate: float = 0.1,
            re_gen_ds_labels=False,
            sampling_method='random',
            num_sample=5,
            walk_length=None,
            bias_strength: float = 2.0,  # 新增：异常感知采样的偏向强度
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

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.name = name  # data name
        self.sampling_method = sampling_method
        self.down_sample_label = dsl
        self.down_sample_rate = down_sample_rate
        self.re_gen_ds_labels = re_gen_ds_labels
        self.num_sample = num_sample
        self.walk_length = walk_length
        self.bias_strength = bias_strength  # 新增
        self.data_dir = data_dir[:-1] if data_dir.endswith('/') else data_dir
        self.mode = kwargs.get('mode', 'feature_only')

        # train/val/test split
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.kwargs = kwargs

    def _generate_cache_filename(self) -> str:
        """生成缓存文件名"""
        # note: 如果mode==feature_only，则数据集名称不显示mode，否则才加入mode，以此来兼容已经创建过的数据集而不必重复计算
        if self.mode == 'feature_only':
            base_params = (
                f'{self.data_dir}/processed/{self.name}'
                f'-method={self.sampling_method}'
            )
        else:
            base_params = (
                f'{self.data_dir}/processed/{self.name}'
                f'-method={self.sampling_method}'
                f'-{self.mode}'
            )

        if self.sampling_method == 'random' or self.sampling_method == 'random_walk':
            filename = (
                f'{base_params}'
                f'-dsl={self.down_sample_label}'
                f'-rate={self.down_sample_rate}'
                f'-walk_length={"+".join(map(str, self.walk_length))}'
                f'-num_sample={self.num_sample}.pkl'
            )
        elif self.sampling_method == 'node2vec':
            filename = (
                f'{base_params}'
                f'-pq={self.kwargs["p"]}+{self.kwargs["q"]}'
                f'-dsl={self.down_sample_label}'
                f'-rate={self.down_sample_rate}'
                f'-walk_length={"+".join(map(str, self.walk_length))}'
                f'-num_sample={self.num_sample}.pkl'
            )
        elif self.sampling_method == 'anomaly_aware':
            # 新增：anomaly_aware方法的文件命名
            filename = (
                f'{base_params}'
                f'-safe_freq=v3'
                f'-bias_strength={self.bias_strength}'
                f'-dsl={self.down_sample_label}'
                f'-rate={self.down_sample_rate}'
                f'-walk_length={"+".join(map(str, self.walk_length))}'
                f'-num_sample={self.num_sample}.pkl'
            )
        else:
            raise ValueError(f'Unknown sampling method: {self.sampling_method}')

        return filename

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            # 生成期望的缓存文件名
            expect_file_name = self._generate_cache_filename()

            # 检查是否使用随机特征
            use_random_feat = self.kwargs.get('random_str_feat', False)

            # 尝试加载缓存的数据集
            if os.path.exists(expect_file_name) and not use_random_feat:
                print(f"Loading cached dataset from: {expect_file_name}")
                dataset = _load_pickle_with_numpy_compat(expect_file_name)
            else:
                print(f"Creating new dataset with method: {self.sampling_method}")

                # 构建数据集参数
                dataset_kwargs = {
                    'name': self.name,
                    'sampling_method': self.sampling_method,
                    'down_sample_label': self.down_sample_label,
                    'down_sample_rate': self.down_sample_rate,
                    're_gen_ds_labels': self.re_gen_ds_labels,
                    'num_sample': self.num_sample,
                    'walk_length': self.walk_length,
                    'seed': 12345,
                    'train_val_test_split': self.hparams.train_val_test_split,
                    'split_shuffle': self.hparams.shuffle,
                    'split_seed': 12345,
                }

                # 为anomaly_aware方法添加特定参数
                if self.sampling_method == 'anomaly_aware':
                    dataset_kwargs['bias_strength'] = self.bias_strength

                # 添加其他kwargs
                dataset_kwargs.update(self.kwargs)

                # 创建数据集
                dataset = AnomalyAwareGDRCDataset(**dataset_kwargs)

                # 保存数据集（如果不使用随机结构特征且采样数不太大）
                if not use_random_feat and self.num_sample <= 30:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(expect_file_name), exist_ok=True)
                    print(f"Saving dataset cache to: {expect_file_name}")
                    with open(expect_file_name, 'wb') as f:
                        pickle.dump(dataset, f)

            # 分割数据集
            self.data_train, self.data_val, self.data_test = split_dataset(
                dataset=dataset,
                frac_list=self.hparams.train_val_test_split,
                shuffle=self.hparams.shuffle,
                random_state=12345  # fix it
            )

        # 统计信息
        self._print_dataset_statistics()

    def _print_dataset_statistics(self):
        """打印数据集统计信息"""
        num_train_anomaly = sum(1 for s in self.data_train if s[1].numpy() == self.down_sample_label)
        num_val_anomaly = sum(1 for s in self.data_val if s[1].numpy() == self.down_sample_label)
        num_test_anomaly = sum(1 for s in self.data_test if s[1].numpy() == self.down_sample_label)

        print(f'''
        ============= Dataset Statistics =============
        Sampling Method: {self.sampling_method}
        {'Bias Strength: ' + str(self.bias_strength) if self.sampling_method == 'anomaly_aware' else ''}

        Train: Normal={len(self.data_train) - num_train_anomaly}, Anomaly={num_train_anomaly}, 
               Total={len(self.data_train)}, Anomaly Rate={num_train_anomaly / len(self.data_train):.2%}

        Val:   Normal={len(self.data_val) - num_val_anomaly}, Anomaly={num_val_anomaly}, 
               Total={len(self.data_val)}, Anomaly Rate={num_val_anomaly / len(self.data_val):.2%}

        Test:  Normal={len(self.data_test) - num_test_anomaly}, Anomaly={num_test_anomaly}, 
               Total={len(self.data_test)}, Anomaly Rate={num_test_anomaly / len(self.data_test):.2%}
        =============================================
        ''')

    def train_dataloader(self) -> GraphDataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return GraphDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> GraphDataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return GraphDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> GraphDataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return GraphDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def get_feature_dim(self) -> int:
        """获取特征维度，便于模型初始化时使用

        :return: 特征维度
        """
        if hasattr(self, 'data_train') and self.data_train is not None:
            sample_graph, _ = self.data_train[0]
            if 'sub_attr' in sample_graph.ndata:
                return sample_graph.ndata['sub_attr'].shape[1]

        # 默认维度估算
        if self.sampling_method == 'anomaly_aware':
            # anomaly_aware方法大约13个特征 * num_sample * len(walk_length)
            walk_length = self.walk_length if self.walk_length else [5, 6, 8, 10]
            return 13 * self.num_sample * len(walk_length)
        else:
            # 原始方法 1个特征 * num_sample * len(walk_length)
            walk_length = self.walk_length if self.walk_length else [5, 6, 8, 10]
            return self.num_sample * len(walk_length)


# 修改数据模块以使用增强版本的数据集
class EnhancedGDRCMDataModule(GDRCMDataModule):
    """增强版本的数据模块，使用保存异常评分的数据集"""

    def setup(self, stage: Optional[str] = None) -> None:
        """加载数据"""
        if not self.data_train and not self.data_val and not self.data_test:

            # 生成期望的缓存文件名 - 添加enhanced标记
            base_filename = self._generate_cache_filename()
            expect_file_name = base_filename.replace('.pkl', '_enhanced.pkl')

            # 检查是否使用随机特征
            use_random_feat = self.kwargs.get('random_str_feat', False)

            # 尝试加载缓存的数据集
            if os.path.exists(expect_file_name) and not use_random_feat:
                print(f"Loading cached enhanced dataset from: {expect_file_name}")
                dataset = _load_pickle_with_numpy_compat(expect_file_name)
            else:
                print(f"Creating new enhanced dataset with method: {self.sampling_method}")

                # 构建数据集参数
                dataset_kwargs = {
                    'name': self.name,
                    'sampling_method': self.sampling_method,
                    'down_sample_label': self.down_sample_label,
                    'down_sample_rate': self.down_sample_rate,
                    're_gen_ds_labels': self.re_gen_ds_labels,
                    'num_walks_per_node': self.num_sample,  # 注意参数名的映射
                    'walk_length': self.walk_length,
                    'seed': self.hparams.seed,
                    'train_val_test_split': self.hparams.train_val_test_split,
                    'split_shuffle': self.hparams.shuffle,
                    'split_seed': 12345,
                    'bias_strength': self.bias_strength,
                    'raw_dir': self.data_dir + '/raw',
                    'save_dir': self.data_dir + '/processed',
                }

                # 添加其他kwargs
                dataset_kwargs.update(self.kwargs)

                # 创建增强版数据集
                dataset = EnhancedSequenceBasedAnomalyAwareDataset(**dataset_kwargs)

                # 保存数据集
                if not use_random_feat and self.num_sample <= 30:
                    os.makedirs(os.path.dirname(expect_file_name), exist_ok=True)
                    print(f"Saving enhanced dataset cache to: {expect_file_name}")
                    with open(expect_file_name, 'wb') as f:
                        pickle.dump(dataset, f)

            # 分割数据集
            from dgl.data.utils import split_dataset
            self.data_train, self.data_val, self.data_test = split_dataset(
                dataset=dataset,
                frac_list=self.hparams.train_val_test_split,
                shuffle=self.hparams.shuffle,
                random_state=12345
            )

        # 统计信息
        self._print_enhanced_dataset_statistics()

    def _print_enhanced_dataset_statistics(self):
        """打印增强数据集的统计信息"""
        # 原始统计
        self._print_dataset_statistics()

        # 额外的异常评分统计
        if hasattr(self.data_train[0][0], 'graph_anomaly_stats'):
            print("\n============= Anomaly Score Statistics =============")

            for split_name, split_data in [('Train', self.data_train),
                                           ('Val', self.data_val),
                                           ('Test', self.data_test)]:
                anomaly_stats = []
                for graph, label in split_data:
                    if hasattr(graph, 'graph_anomaly_stats'):
                        anomaly_stats.append(graph.graph_anomaly_stats['mean_anomaly'])

                if anomaly_stats:
                    mean_anomaly = np.mean(anomaly_stats)
                    std_anomaly = np.std(anomaly_stats)
                    print(f"{split_name}: Mean graph anomaly score = {mean_anomaly:.4f} ± {std_anomaly:.4f}")

            print("===================================================\n")
