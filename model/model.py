from utils import load_model, preprocess_for_model
import numpy as np


class Model:
    # load model from a given path
    def __init__(self, path: str):
        # load saved model to use
        self.model = load_model(path)

    def predict_batch(self, images):
        processed = [preprocess_for_model(img) for img in images]
        processed = np.squeeze(processed, axis=1)
        preds = self.model.predict(images)
        return preds.argmax(axis=-1)

    def predict(self, image):
        result = self.model.predict(image)
        return result.argmax(axis=-1)
