import argparse
import os
import logging
import configparser
import pandas as pd

from model import BBCModel


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train(config_path: str):
    logger.debug("Parse config...")
    config = configparser.ConfigParser()
    config.read(config_path)

    model = BBCModel(config["MODEL"])

    experiments_dir = config["PREPROCESSOR"]["experiments_dir"]
    train_split_path = os.path.join(experiments_dir, "BBC_News_Train_Split.csv")
    train_data = pd.read_csv(train_split_path)

    val_split_path = os.path.join(experiments_dir, "BBC_News_Val_Split.csv")
    val_data = pd.read_csv(val_split_path)

    model.fit(train_data.Text, train_data.Category)
    pred = model.predict(train_data.Text)
    acc = (pred == train_data.Category).sum() / len(pred)
    logger.debug(f"TRAIN_ACC: {acc}")

    pred = model.predict(val_data.Text)
    acc = (pred == val_data.Category).sum() / len(pred)
    logger.debug(f"VAL_ACC: {acc}")

    model.save()


def parse_args():
    default_path = "/home/dkrivenkov/program/mipt_mle/news_classification/configs/train_config.ini"
    parser = argparse.ArgumentParser(description="predict script")
    parser.add_argument("--config_path", type=str, default=default_path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args.config_path)
