import argparse
import logging
import json
import os
import configparser
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCCPreprocessor:
    def __init__(self, config_path) -> None:
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def load_train_data(self):
        data = pd.read_csv(self.config["PREPROCESSOR"]["data_path_train"])
        return data

    def load_test_data(self):
        data = pd.read_csv(self.config["PREPROCESSOR"]["data_path_test"])
        return data

    def target_transform(self, data, mode: str = "train"):
        if mode == "train":
            unique_category = np.unique(data.Category)
            self.category2id = dict(zip(
                unique_category,
                range(len(unique_category))
            ))
        data.Category = data.Category.map(lambda x: self.category2id[x])

    def split_data(self, data: pd.DataFrame):
        test_size = self.config["PREPROCESSOR"].getfloat("test_size")
        random_state = self.config["PREPROCESSOR"].getint("random_state")

        np.random.seed(random_state)
        indices = np.random.permutation(range(len(data)))
        test_count = int(len(data) * test_size)

        test_idxs = indices[: test_count]
        train_idx = indices[test_count:]

        return data.iloc[train_idx], data.iloc[test_idxs]

    def save_data(self, data: pd.DataFrame, mode: str = "train"):
        experiments_dir = self.config["PREPROCESSOR"]["experiments_dir"]
        if not os.path.exists(experiments_dir):
            os.mkdir(experiments_dir)

        if mode == "train":
            train_split_path = os.path.join(
                experiments_dir,
                "BBC_News_Train_Split.csv"
            )
            logger.info(f"Try to save data to: {train_split_path}")
            data.to_csv(train_split_path, index=False)
        elif mode == "val":
            val_split_path = os.path.join(
                experiments_dir,
                "BBC_News_Val_Split.csv"
            )
            logger.info(f"Try to save data to: {val_split_path}")
            data.to_csv(val_split_path, index=False)
        elif mode == "test":
            test_split_path = os.path.join(
                experiments_dir,
                "BBC_News_Test_Split.csv"
            )
            logger.info(f"Try to save data to: {test_split_path}")
            data.to_csv(test_split_path, index=False)
        else:
            raise ValueError(f"Current mode {mode} not exist")

    def save_metadata(self):
        with open(self.config["PREPROCESSOR"]["target_mapping"], "w") as fp:
            json.dump(self.category2id, fp)


def parse_args():
    default_path = "./configs/train_config.ini"
    parser = argparse.ArgumentParser(description="predict script")
    parser.add_argument("--config_path", type=str, default=default_path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    preprocessor = BCCPreprocessor(args.config_path)

    bbc_train_data = preprocessor.load_train_data()
    bbc_test_data = preprocessor.load_test_data()

    preprocessor.target_transform(bbc_train_data, mode="train")
    bbc_train, bbc_val = preprocessor.split_data(bbc_train_data)

    preprocessor.save_data(bbc_train, mode="train")
    preprocessor.save_data(bbc_val, mode="val")
    preprocessor.save_data(bbc_test_data, mode="test")
    preprocessor.save_metadata()
