import os
from typing import Callable, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import CorpusDataset
from net import SpacingBertModel

NUM_CORES=os.cpu_count()

def get_dataloader(
    data_path: str, transform: Callable[[List, List], Tuple], batch_size: int
) -> DataLoader:
    
    dataset = CorpusDataset(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=NUM_CORES)
    return dataloader


def main(config):
    preprocessor = Preprocessor(config.max_len)
    train_dataloader = get_dataloader(
        config.train_data_path, preprocessor.get_input_features, config.train_batch_size
    )
    val_dataloader = get_dataloader(
        config.val_data_path, preprocessor.get_input_features, config.train_batch_size
    )
    test_dataloader = get_dataloader(
        config.test_data_path, preprocessor.get_input_features, config.eval_batch_size
    )

    bert_finetuner = SpacingBertModel(
        config, train_dataloader, val_dataloader, test_dataloader
    )

    logger = TensorBoardLogger(save_dir=config.log_path, version=1, name=config.task)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/{epoch}_{val_loss:3f}",
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        gpus=config.gpus,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        strategy='dp',
        max_epochs=config.epochs
    )

    trainer.fit(bert_finetuner)
    trainer.test()


if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    main(config)
