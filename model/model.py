from utils import load_model, preprocess_for_model


class Model:
    # load model from a given path
    def __init__(self, path: str):
        # load saved model to use
        self.model = load_model(path)

    def predict(self, image):
        # customize image to predict
        image = preprocess_for_model(image)
        return self.model.predict(image).argmax()
