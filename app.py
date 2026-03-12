import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("iris_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    features=[float(x) for x in request.form.values()]
    final=[np.array(features)]

    prediction=model.predict(final)

    species=["Setosa","Versicolor","Virginica"]

    result=species[prediction[0]]

    return render_template(
        "index.html",
        prediction_text="Flower species is {}".format(result)
    )

if __name__=="__main__":
    app.run(host="0.0.0.0",port=10000)