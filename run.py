from flask import Flask, request, jsonify
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

app = Flask(__name__)

@app.route('/predict')
def predict():
    json_path = request.args.get('json_path')
    model_path = request.args.get('model_path')

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    predictor = ModelPredictor(model_path)
    probs = predictor.predict_scores(data)
    
    return jsonify({"score": probs})

@app.route('/')
def home():
    return jsonify({"a": "a"});

if __name__ == '__main__':
    app.run(debug=True)