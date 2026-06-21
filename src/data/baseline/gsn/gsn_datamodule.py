# %%
from typing import Any, Dict, Optional, Tuple

from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader
from lightning import LightningDataModule
from torch.utils.data import Dataset

from src.data.baseline.gsn.gsn_dataset2 import GSNDataset


class GSNDataModule(LightningDataModule):
    def __init__(
            self,
            name: str = 'AIDS',
            dsl: int = 0,
            down_sample_rate: float = 0.1,
            default_feat_dim: int = -1,
            re_gen_ds_labels=False,
            induced=False,
            is_directed=False,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (0.7, 0.2, 0.1),
            shuffle: bool = False,
            seed: int = 12345,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.name = name  # data name
        self.down_sample_label = dsl
        self.down_sample_rate = down_sample_rate
        self.re_gen_ds_labels = re_gen_ds_labels
        self.default_feat_dim = default_feat_dim
        self.induced = induced
        self.is_directed = is_directed

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = GSNDataset(name=self.name,
                                 down_sample_label=self.down_sample_label,
                                 down_sample_rate=self.down_sample_rate,
                                 re_gen_ds_labels=self.re_gen_ds_labels,
                                 default_feat_dim=self.default_feat_dim,
                                 induced=self.induced,
                                 is_directed=self.is_directed,
                                 seed=self.hparams.seed)
            # split dataset
            self.data_train, self.data_val, self.data_test = split_dataset(
                dataset=dataset,
                frac_list=self.hparams.train_val_test_split,
                shuffle=self.hparams.shuffle,
                random_state=12345  # we fix it to 12345
            )
            # do some statistical things
            num_train_anomaly = 0
            num_val_anomaly = 0
            num_test_anomaly = 0
            for s in self.data_train:
                if s[1].numpy() == self.down_sample_label:
                    num_train_anomaly += 1
            for s in self.data_val:
                if s[1].numpy() == self.down_sample_label:
                    num_val_anomaly += 1
            for s in self.data_test:
                if s[1].numpy() == self.down_sample_label:
                    num_test_anomaly += 1
            print('''
                    -------------
                    train: G0={}, G1={}, All: {}, anomaly rate={:.2%}
                    val G0={}, G1={}, All: {}, anomaly rate={:.2%}
                    test G0={}, G1={}, All: {}, anomaly rate={:.2%}
                    -------------
                    '''.format(
                # train
                num_train_anomaly,
                len(self.data_train) - num_train_anomaly,
                len(self.data_train),
                num_train_anomaly / len(self.data_train),
                # val
                num_val_anomaly,
                len(self.data_val) - num_val_anomaly,
                len(self.data_val),
                num_val_anomaly / len(self.data_val),
                # test
                num_test_anomaly,
                len(self.data_test) - num_test_anomaly,
                len(self.data_test),
                num_test_anomaly / len(self.data_test)
            ))

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


if __name__ == "__main__":
    datamodule = GSNDataModule()
    datamodule.setup()
    print(datamodule.train_dataloader())
    print(datamodule.val_dataloader())
    print(datamodule.test_dataloader())
