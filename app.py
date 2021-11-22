from flask import Flask,render_template,request
import pickle
import sklearn
import pandas as pd
import numpy as np


app = Flask(__name__)
model = pickle.load(open('trained_SVM_model.pkl', 'rb'))
scaler = pickle.load(open('normalizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    print(output)  

    if output == 1:
        return render_template('predict.html', prediction='AFFECTED')
    else:
        return render_template('predict.html', prediction = 'Not_affected')


if __name__ == '__main__':
    app.run()