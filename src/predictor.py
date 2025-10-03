import pickle
import numpy as np

def load_model(model_path="models/digit_model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_model(model, sample):
    sample = np.array(sample).reshape(1, -1)  # Ensure correct shape
    return model.predict(sample)[0]
