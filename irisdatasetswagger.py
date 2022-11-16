# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:52:52 2022

@author: kirub
"""

import pickle
from flask import Flask, request
import pandas as pd
import numpy as np
import flasgger
from flasgger import Swagger

with open('model_svm.pkl','rb') as model_svm_pickle:
    model_svm = pickle.load(model_svm_pickle)
    
with open('model_rf.pkl','rb') as model_rf_pickle:
    model_rf = pickle.load(model_rf_pickle)
    
ml_api = Flask(__name__)
swagger = Swagger(ml_api)


@ml_api.route('/predict_svc', methods=['Get'])
def predict_svc():    
    """Let's classify iris dataset [0:'setosa', 1:'versicolor', 2:'verginica']
    Species Classification
    ---
    parameters:  
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    sepal_length = request.args.get('sepal_length')
    sepal_width = request.args.get('sepal_width')
    petal_length = request.args.get('petal_length')
    petal_width = request.args.get('petal_width')
    
    input_data = np.array([[sepal_length, sepal_width,petal_length,petal_width]])
    prediction = model_svm.predict(input_data)
    return str(prediction)

@ml_api.route('/predict_sv_file', methods=['POST'])
def predict_sv_file():
    """Let's classify iris dataset 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true 
    responses:
        200:
            description: The output values
      
            
    """
    input_data = pd.read_csv(request.files.get("file"))
    prediction = model_svm.predict(input_data)
    return str(list(prediction))

if __name__ == '__main__':
    ml_api.run()