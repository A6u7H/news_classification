import logging
import os
import pickle

from sklearn.svm import SVC
from omegaconf import DictConfig
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class BBCModel:
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = self.create_model()
        self.vectorizer = self.create_vectorizer()

    def create_vectorizer(self):
        return TfidfVectorizer()

    def create_model(self):
        logger.debug(f"Creating model: {self.config['type']}...")
        if self.config["type"] == "svm":
            classifier = SVC(
                kernel=self.config["kernel"], 
                random_state=self.config.getint("random_state")
            )
        else:
            raise ValueError(f"Model type {self.config['type']} does not exist")
        logger.debug("Model has created")
        return classifier

    def fit(self, texts, target):
        features = self.vectorizer.fit_transform(texts)
        logger.debug(f"Starting train model: {self.config['type']}...")
        self.model.fit(features, target)
        logger.debug("Model has trained")

    def predict(self, texts):
        features = self.vectorizer.transform(texts)
        logger.debug(f"Predicting: {self.config['type']}...")
        return self.model.predict(features)

    def save(self, save_folder: Optional[str]=None) -> None:
        if save_folder is None:
            save_folder = self.config["path"]

        vectorizer_path = os.path.join(save_folder, "vectorizer.pickle")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        model_path = os.path.join(save_folder, f"{self.config['type']}.sav")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.debug("Model and config have saved")

    def load(self, load_folder: Optional[str]=None) -> None:
        if load_folder is None:
            load_folder = self.config["path"]

        model_path = os.path.join(load_folder, f"{self.config['type']}.sav")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        vectorizer_path = os.path.join(load_folder, "vectorizer.pickle")
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)


