import configparser
import os
import unittest
import sys

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.join(os.getcwd(), "src"))
from model import BBCModel


config_path = "./configs/train_config.ini"


class TestDataMaker(unittest.TestCase):
    def setUp(self) -> None:
        config = configparser.ConfigParser()
        config.read(config_path)
        self.model = BBCModel(config["MODEL"])

    def test_model_type(self):
        self.assertEqual(type(self.model.model), SVC)

    def test_vectorizer_type(self):
        self.assertEqual(type(self.model.vectorizer), TfidfVectorizer)


if __name__ == "__main__":
    unittest.main()
