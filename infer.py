import argparse
import json
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class ModelPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_scores(self, data_dict):
        data = np.array([list(data_dict.values())]).reshape(1, -1)
        probs = self.model.predict_proba(data)
        out = probs[0][-1] * 100
        return out

def main():
    parser = argparse.ArgumentParser(description='Predict scores using a trained model.')
    parser.add_argument('--file_path', type=str, help='Path to the JSON file containing the data points')
    parser.add_argument('--weights', type=str, help='Path to the saved model weights file (e.g., pkl file)')
    args = parser.parse_args()

    with open(args.file_path, 'r') as json_file:
        data = json.load(json_file)

    predictor = ModelPredictor(args.weights)
    probs = predictor.predict_scores(data)
    print("Score:", probs)

if __name__ == "__main__":
    main()