from flask import Flask
from flask import request
import pickle
import pandas as pd
import json
from io import BytesIO

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open("models/lg_model.pkl", "rb"))
    data_test = pd.read_csv('Data/test.csv', index_col='Id')
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



@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run("0.0.0.0")




##@app.route("https://site/api/predict/",methods=['POST'])
##def predict():
##    return "/models/lg_model.pkl"


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
