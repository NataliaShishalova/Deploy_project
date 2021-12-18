from flask import Flask
from flask import request
import json

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open("models/lg_model.pkl", "rb"))
    prediction = model.predict("data/test.scv")
    return prediction

#@app.route('/model_name/predict', methods=['GET'])
#def model_name_predict():
#    model = pickle.load(open("models/lg_model.pkl", "rb"))
#    prediction = model.predict("data/test.scv")
#    return prediction

@app.route('/')
def index()
    return "Hello, World!"

if __name__ == '__main__':
    app.run()




##@app.route("https://site/api/predict/",methods=['POST'])
##def predict():
##    return "/models/lg_model.pkl"


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
