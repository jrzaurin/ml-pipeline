from __future__ import print_function

import os
import json
import pickle
import io
import sys
import signal
import traceback

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# This model filename MUST match the filename used when saving it after the training
MODEL_FILENAME = "lightgbm-0.pkl"

class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, MODEL_FILENAME), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        print("---JSON")
        data = flask.request.data.decode('utf-8')
        print(data)
        print("....")
        s = io.StringIO(data)
        data = pd.read_json(s)
        print(data)
        print("....")
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')
        
    print('Invoked with {} records'.format(data.shape[0]))
    dataprocessor_fname = 'dataprocessor_0_.p'
    dataprocessor = pickle.load(open('/opt/ml/{}'.format(dataprocessor_fname), 'rb'))

    # Drop first column, since sample notebook uses training data to show case predictions
    data.drop(data.columns[[0]],axis=1,inplace=True)
    trow = dataprocessor.transform(data)

    # Do the prediction
    predictions = ScoringService.predict(trow)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
