import os
import json
import logging
import numpy as np
import pandas as pd
import configparser

from omegaconf import DictConfig

from model import BBCModel
from preprocessor import BCCPreprocessor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def predict(config_path: str):
    logger.debug("Parse config...")
    config = configparser.ConfigParser()
    config.read(config_path)

    model = BBCModel(config["MODEL"])
    model.load()

    test_data = pd.read_csv(config["PREPROCESSOR"]["test_preprocessed"])

    with open(config["PREPROCESSOR"]["target_mapping"], "r") as fp:
        category2id = json.load(fp)

    id2category = {v : k for k, v in category2id.items()}

    logger.debug("Starting predict stage")
    pred = model.predict(test_data.Text)

    func = np.vectorize(lambda x: id2category[x])
    pred_category = func(pred)

    solution = pd.DataFrame(
        data={
            "ArticleId" : test_data.ArticleId.values,
            "Category": pred_category
        }
    )
    solution.to_csv(config["SOLUTION"]["save_path"], index=False)

if __name__ == "__main__":
    config_path = "/home/dkrivenkov/program/mipt_mle/news_classification/configs/train_config.ini"
    predict(config_path)