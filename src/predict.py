import json
import os
import logging
import argparse
import numpy as np
import pandas as pd
import configparser

from model import BBCModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(config_path: str):
    logger.info("Parse config...")
    config = configparser.ConfigParser()
    config.read(config_path)

    model = BBCModel(config["MODEL"])
    model.load()

    experiments_dir = config["PREPROCESSOR"]["experiments_dir"]
    test_data_path = os.path.join(experiments_dir, "BBC_News_Test_Split.csv")
    test_data = pd.read_csv(test_data_path)

    with open(config["PREPROCESSOR"]["target_mapping"], "r") as fp:
        category2id = json.load(fp)

    id2category = {
        v: k for k, v in category2id.items()
    }

    logger.info("Starting predict stage")
    pred = model.predict(test_data.Text)

    func = np.vectorize(lambda x: id2category[x])
    pred_category = func(pred)

    solution = pd.DataFrame(
        data={
            "ArticleId": test_data.ArticleId.values,
            "Category": pred_category
        }
    )
    solution.to_csv(config["SOLUTION"]["save_path"], index=False)


def parse_args():
    default_path = "./configs/train_config.ini"
    parser = argparse.ArgumentParser(description="predict script")
    parser.add_argument("--config_path", type=str, default=default_path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args.config_path)
