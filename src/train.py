import argparse
import os
import logging
import configparser
import pandas as pd

from model import BBCModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config_path: str) -> None:
        logger.debug("Parse config...")
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.model = BBCModel(self.config["MODEL"])

    def fit(self):
        experiments_dir = self.config["PREPROCESSOR"]["experiments_dir"]
        train_split_path = os.path.join(
            experiments_dir,
            "BBC_News_Train_Split.csv"
        )
        train_data = pd.read_csv(train_split_path)

        val_split_path = os.path.join(experiments_dir, "BBC_News_Val_Split.csv")
        val_data = pd.read_csv(val_split_path)

        self.model.fit(train_data.Text, train_data.Category)
        pred = self.model.predict(train_data.Text)
        acc = (pred == train_data.Category).sum() / len(pred)
        logger.debug(f"TRAIN_ACC: {acc}")

        pred = self.model.predict(val_data.Text)
        acc = (pred == val_data.Category).sum() / len(pred)
        logger.debug(f"VAL_ACC: {acc}")

        self.model.save()


def parse_args():
    default_path = "./configs/train_config.ini"
    parser = argparse.ArgumentParser(description="predict script")
    parser.add_argument("--config_path", type=str, default=default_path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args.config_path)
    trainer.fit()
