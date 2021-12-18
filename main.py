from flask import Flask
from flask import request
import pickle
import pandas as pd
import json
from io import BytesIO
from os import listdir
from os.path import isfile, join
from pathlib import Path

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open("models/lg_model.pkl", "rb"))
    data_test = pd.read_csv('data/test.csv', index_col='Id')
    test_features = data_test.drop(columns=['Open Date', 'City', 'City Group', 'Type'])
    prediction = model.predict(test_features).tolist()
    return json.dumps(prediction)

@app.route('/predict_2', methods=['POST'])
def predict_2():
    data_io = BytesIO(request.data)
    data_test = pd.read_csv(data_io, index_col='Id')

    test_features = data_test.drop(columns=['Open Date', 'City', 'City Group', 'Type'])
    model = pickle.load(open("models/lg_model.pkl", "rb"))

    prediction = model.predict(test_features).tolist()

    return json.dumps(prediction)


@app.route('/predict_model_name', methods=['POST'])
def predict_model_name():
    model_name = request.args.get('name')

    data_io = BytesIO(request.data)
    data_test = pd.read_csv(data_io, index_col='Id')
    test_features = data_test.drop(columns=['Open Date', 'City', 'City Group', 'Type'])

    model = pickle.load(open("models/" + model_name + ".pkl", "rb"))

    prediction = model.predict(test_features).tolist()

    return json.dumps(prediction)

@app.route('/model_list', methods=['GET'])
def model_list():
    m_list = [Path(f).stem for f in listdir("models") if isfile(join("models", f))]
    return json.dumps(m_list)

if __name__ == '__main__':
    app.run("0.0.0.0")




##@app.route("https://site/api/predict/",methods=['POST'])
##def predict():
##    return "/models/lg_model.pkl"


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
