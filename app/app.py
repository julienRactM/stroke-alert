from flask import Flask, render_template, request
import joblib
import pandas as pd
from script import model_prediction
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    titles = []

    titles = "Allez vous subir un AVC ?"

    if request.method == 'POST':
        # model = joblib.load('app/app_model/current_model.pkl')
        age = request.form.get('age')
        hypertension = request.form.get('hypertension')
        avg_glucose_level = request.form.get('avg_glucose_level')
        heart_disease = request.form.get('heart_disease')
        smoking_status = request.form.get('smoking_status')
        X_test = pd.DataFrame({'age'            : [age],
                        'hypertension'       : [hypertension],
                        'avg_glucose_level'  : [avg_glucose_level],
                        'heart_disease' : [heart_disease],
                        'smoking_status' : [smoking_status]
                        })

        prediction = model_prediction(age, hypertension, avg_glucose_level, heart_disease, smoking_status)[0]
        predict_proba = model_prediction(age, hypertension, avg_glucose_level, heart_disease, smoking_status)[1]

        if prediction == 1 :
            comment = "L'algorithme détecte un risque d'AVC"
        else :
            comment = "L'algorithme ne prédit pas de risque accru d'AVC"

        return render_template('index.html', titles = titles, prediction=prediction, comment=comment, age=age,\
            hypertension=hypertension, avg_glucose_level=avg_glucose_level, heart_disease=heart_disease, smoking_status=smoking_status,\
                predict_proba = np.round(predict_proba[0][1], 2)) # active_tab='home'


    return render_template('index.html') # active_tab='home'

if __name__ == '__main__':
    app.run(debug=True)
