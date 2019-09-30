from flask import Flask, request, jsonify
from flask_restful import reqparse, abort, Api, Resource

import numpy as np
import joblib

app = Flask(__name__)




@app.route('/',methods=['GET'])
def home():
    return '<h1>Welcome to our ML api of Abusive Words Detection</h1>'

vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('mymodel.joblib')


def probability(prob):
  return prob[1]


def predict(texts):
  return model.predict(vectorizer.transform(texts))

def predict_prob(texts):
  return np.apply_along_axis(probability, 1, model.predict_proba(vectorizer.transform(texts)))

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

@app.route('/predict', methods=['POST'])
def predicttext():

    args = parser.parse_args()
    querytext = args['query']
    prediction=predict([querytext])
    prob=predict_prob([querytext])

    if prediction == 0:
        pred_text = 'Positive'
        value = 0
    else:
        pred_text = 'Negative'
        value =1

 
    output = {'Abusive': value,'prediction': pred_text,'status':200}
    return output



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) 
    except:
        port = 5000


  
    app.run(port=port, debug=True)
