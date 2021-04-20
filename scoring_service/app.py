"""
This module defines the scoring service in the following steps:

- loads the ML model into memory;
- defines the ML scoring REST API endpoints; and,
- starts the service.

"""
from typing import Dict
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests
import numpy as np
from flask import Flask, jsonify, make_response, request, Response
from joblib import load
from sklearn.base import BaseEstimator
from . application_score import compute_prediction

MODEL_PATH = 'XGBoost.sav'

app = Flask(__name__)


@app.route('/api/v1/app_scoring/score', methods=['POST'])
def score() -> Response:
    """Application scoring API endpoint"""
    request_data = request.json

    request_data     = request.get_json()
    value1           = request_data.get('Final branch')
    value2           = request_data.get('principal_amount')
    if value1 is not None and value2 is not None:
        model_output = compute_prediction(request_data)
    response_data = jsonify({**model_output, 'model_info': str(model)})
    return make_response(response_data)



# if __name__ == '__main__':
#     model = pickle.load(open(MODEL_PATH, 'rb'))
#     print(f'loaded model={model}')
#     print(f'starting API server')
#     app.run(host='0.0.0.0')
