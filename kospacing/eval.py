import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import CorpusDataset
from net import SpacingBertModel
import warnings
warnings.filterwarnings("ignore")
NUM_CORES=os.cpu_count()

def get_dataloader(data_path, transform, batch_size):
    dataset = CorpusDataset(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=NUM_CORES)
    return dataloader


def main(config):

    preprocessor = Preprocessor(config.max_len)
    test_dataloader = get_dataloader(
        config.test_data_path, preprocessor.get_input_features, config.eval_batch_size
    )
    model = SpacingBertModel(config, None, None, test_dataloader)

    checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer()
    return trainer.test(model)



if __name__ == "__main__":
    config = OmegaConf.load("config/eval_config.yaml")
    main(config)
