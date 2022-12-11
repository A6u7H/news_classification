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

    train_data = pd.read_csv(config["PREPROCESSOR"]["train_preprocessed"])
    val_data = pd.read_csv(config["PREPROCESSOR"]["val_preprocessed"])

    model.fit(train_data.Text, train_data.Category)
    pred = model.predict(train_data.Text)
    acc = (pred == train_data.Category).sum() / len(pred)
    logger.debug(f"TRAIN_ACC: {acc}")

    pred = model.predict(val_data.Text)
    acc = (pred == val_data.Category).sum() / len(pred)
    logger.debug(f"VAL_ACC: {acc}")

    model.save()

if __name__ == "__main__":
    config_path = "/home/dkrivenkov/program/mipt_mle/news_classification/configs/train_config.ini"
    train(config_path)