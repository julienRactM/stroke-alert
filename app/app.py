from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    titles = []

    titles = "Allez vous subir un AVC ?"

    if request.method == 'POST':
        model = joblib.load('app/app_model/model.pkl')
        gender = request.form.get('gender')
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        Residence_type = request.form.get('Residence_type')
        smoking_status = request.form.get('smoking_status')
        X_test = pd.DataFrame({'gender'         : [gender],
                               'age'            : [age],
                               'bmi'            : [bmi],
                               'Residence_type' : [Residence_type],
                               'smoking_status' : [smoking_status]
                               })



        prediction = model.predict(X_test)
        if prediction == 1 :
            comment = "Oh, je vois, comment ça sent pas bon pour toi"
        else :
            comment = "Félicitation tu devrais vivre encore un peu, on se trompe à 6% quand même fait pas le fou"

        return render_template('index.html', titles = titles, prediction=prediction, comment=comment, gender=gender, \
            age=age, bmi=bmi, Residence_type=Residence_type, smoking_status=smoking_status) # active_tab='home'


    return render_template('index.html') # active_tab='home'

if __name__ == '__main__':
    app.run(debug=True)
