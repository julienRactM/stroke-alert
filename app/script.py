import sys
import pickle
import pandas as pd


def model_predict(age, hypertension, avg_glucose_level, heart_disease, smoking_status):
    # model = pickle.load('app/app_model/best_rf_model.pkl')
    with open('app/app_model/best_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    X_test = pd.DataFrame({'age'            : [age],
                       'hypertension'       : [hypertension],
                       'avg_glucose_level'  : [avg_glucose_level],
                       'heart_disease' : [heart_disease],
                       'smoking_status' : [smoking_status]
                       })

    # return model.predict(X_test)

def error_message():
    print("""Il faut entrer en ligne de commande :
        python script.py age(float range[10-120]) hypertension(int 1=True) avg_glucose_level(float range [50 300]) heart_disease(int 1=True) smoking_status(str smokes - formerly smoked - never smoked)""")
    return

def main():

    if len(sys.argv) == 6:
        age = float(sys.argv[1])
        hypertension = int(sys.argv[2])
        avg_glucose_level = float(sys.argv[3])
        heart_disease = int(sys.argv[4])
        smoking_status = sys.argv[5].strip()
    else :
        error_message()
        return

    prediction = model_predict(age, hypertension, avg_glucose_level, heart_disease, smoking_status)

    if prediction == 0:
        return print("Pas de risque accru d'avc détecté")
    elif prediction == 1:
        return print("Risque d'avc détecté")
    else:
        return print("Invalid prediction")
    # return prediction



if __name__ == "__main__":
    main()
