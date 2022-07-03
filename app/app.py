from flask import Flask, render_template, request
from joblib import load
import numpy as np


app = Flask(__name__)

model = load('app/disease_pred.joblib')


@app.route("/")
def hello_world():
        return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
        int_inputs = [int(x) for x in request.form.values()]
        inputs = [np.array(int_inputs)]
        preds = model.predict(inputs)

        return render_template('index.html', prediction_text = "The Likely Disease is: {}".format(preds))



if __name__=="__main__":
    app.run(debug=True)
