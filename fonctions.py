import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.stats import chi2_contingency
import lime
import lime.lime_tabular
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve,make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def impute_and_plot(df_0, df_1,df):

    categorical_columns = df_0.select_dtypes(include=['object']).columns


    encoder = OneHotEncoder( sparse=False)
    df_encoded_temp_0 = pd.DataFrame(encoder.fit_transform(df_0[categorical_columns]),
                                     columns=encoder.get_feature_names_out(categorical_columns),
                                     index=df_0.index)

    df_encoded_temp_1 = pd.DataFrame(encoder.transform(df_1[categorical_columns]),
                                     columns=encoder.get_feature_names_out(categorical_columns),
                                     index=df_1.index)


    df_encoded_0 = pd.concat([df_0.drop(categorical_columns, axis=1), df_encoded_temp_0], axis=1)
    df_encoded_1 = pd.concat([df_1.drop(categorical_columns, axis=1), df_encoded_temp_1], axis=1)


    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn_imputed_0 = knn_imputer.fit_transform(df_encoded_0)
    df_knn_imputed_1 = knn_imputer.transform(df_encoded_1)


    median_imputer = SimpleImputer(strategy='median')
    df_median_imputed_0 = median_imputer.fit_transform(df_encoded_0)
    df_median_imputed_1 = median_imputer.transform(df_encoded_1)


    mean_imputer = SimpleImputer(strategy='mean')
    df_mean_imputed_0 = mean_imputer.fit_transform(df_encoded_0)
    df_mean_imputed_1 = mean_imputer.transform(df_encoded_1)


    df_knn_imputed = pd.concat([pd.DataFrame(df_knn_imputed_0, columns=df_encoded_0.columns),
                                pd.DataFrame(df_knn_imputed_1, columns=df_encoded_1.columns)])

    df_median_imputed = pd.concat([pd.DataFrame(df_median_imputed_0, columns=df_encoded_0.columns),
                                   pd.DataFrame(df_median_imputed_1, columns=df_encoded_1.columns)])

    df_mean_imputed = pd.concat([pd.DataFrame(df_mean_imputed_0, columns=df_encoded_0.columns),
                                 pd.DataFrame(df_mean_imputed_1, columns=df_encoded_1.columns)])


    plt.figure(figsize=(12, 8))


    plt.subplot(2, 2, 1)
    sns.histplot(df['bmi'].dropna(), kde=True, color='blue', label='Avant Imputation')
    plt.title('Distribution de bmi avant Imputation')
    plt.xlabel('bmi')
    plt.legend()
    skewness_before = df_0['bmi'].dropna().skew()
    kurt_before = df_0['bmi'].dropna().kurt()
    plt.text(0.97, 0.80, s=f"Skewness: {skewness_before:.2f}\nKurtosis: {kurt_before:.2f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', backgroundcolor='white')


    plt.subplot(2, 2, 2)
    sns.histplot(df_knn_imputed['bmi'], kde=True, color='green', label='Après Imputation (KNN)')
    plt.title('Distribution de bmi après Imputation (KNN)')
    plt.xlabel('bmi')
    plt.legend()
    skewness_knn = df_knn_imputed['bmi'].skew()
    kurt_knn = df_knn_imputed['bmi'].kurt()
    plt.text(0.97, 0.80, s=f"Skewness: {skewness_knn:.2f}\nKurtosis: {kurt_knn:.2f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', backgroundcolor='white')


    plt.subplot(2, 2, 3)
    sns.histplot(df_median_imputed['bmi'], kde=True, color='orange', label='Après Imputation (Médiane)')
    plt.title('Distribution de bmi après Imputation (Médiane)')
    plt.xlabel('bmi')
    plt.legend()
    skewness_median = df_median_imputed['bmi'].skew()
    kurt_median = df_median_imputed['bmi'].kurt()
    plt.text(0.97, 0.80, s=f"Skewness: {skewness_median:.2f}\nKurtosis: {kurt_median:.2f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', backgroundcolor='white')

 
    plt.subplot(2, 2, 4)
    sns.histplot(df_mean_imputed['bmi'], kde=True, color='purple', label='Après Imputation (Moyenne)')
    plt.title('Distribution de bmi après Imputation (Moyenne)')
    plt.xlabel('bmi')
    plt.legend()
    skewness_mean = df_mean_imputed['bmi'].skew()
    kurt_mean = df_mean_imputed['bmi'].kurt()
    plt.text(0.97, 0.80, s=f"Skewness: {skewness_mean:.2f}\nKurtosis: {kurt_mean:.2f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', backgroundcolor='white')

    plt.tight_layout()
    
    plt.show()
    return df_knn_imputed

#######################################

def preprocessing(df, standardiser=StandardScaler(), mode='oversampling',nbr_feature = 19):
    cat_cols = df.select_dtypes(exclude=['float']).columns
    num_cols = df.select_dtypes(include=['float']).columns
    
    df_cat_encoded = pd.get_dummies(df[cat_cols], drop_first=False)
    
    if not num_cols.empty:  
        df_encoded = pd.concat([df_cat_encoded, df[num_cols]], axis=1)
    else:
        df_encoded = df_cat_encoded
    
    X = df_encoded.drop('stroke',axis=1)
    y = df['stroke']
    
    selector = SelectKBest(k=nbr_feature)  
    X_selected = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    
    df_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    X_train, X_test, y_train, y_test = train_test_split(df_selected, y, test_size=0.2, random_state=42)
    
    if mode == 'undersampling':
        rus = RandomUnderSampler(sampling_strategy='auto')
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif mode == 'oversampling':
        ros = RandomOverSampler(sampling_strategy='auto')
        X_train, y_train = ros.fit_resample(X_train, y_train)
    elif mode is None:
        X_train, y_train = X_train, y_train
    
    if not num_cols.empty: 
        X_train[num_cols] = standardiser.fit_transform(X_train[num_cols])
        X_test[num_cols] = standardiser.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, selected_features
#######################################

def modele_dummy(X_train,y_train,X_test,y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent",random_state=0)
    dummy_clf.fit(X_train, y_train)
    y_pred_dummy = dummy_clf.predict(X_test)
    y_pred_proba_dummy = dummy_clf.predict_proba(X_test)[:, 1]
    dummy_score = dummy_clf.score(X_train,y_train)
    
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_dummy)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_dummy = (y_pred_proba_dummy>= optimal_threshold).astype(int)
    
    print('Scores du modèle ')
    print(evaluate_classification_model(y_test, y_pred_dummy, y_pred_proba_dummy))
    results = evaluate_classification_model(y_test, y_pred_dummy, y_pred_proba_dummy)    
    print('Affichage de la courbe Roc ')
    print(create_roc_auc_plot(dummy_clf,y_test, y_pred_dummy))
    
    print('Affichage du rapport de classification ')
    print(create_classification_report(dummy_clf,y_test,y_pred_dummy))
    
    print('Affichage de la matrice de confusion')
    print(create_confusion_matrix_plot(dummy_clf, y_test, y_pred_dummy))
    
    
    return dummy_clf, y_pred_dummy,y_pred_proba_dummy, results

#######################################

def random_forest(X_train,y_train,X_test,y_test,x_col,mode='undersampling',standardiser='RobustScaler',feature='Hypertension'):
    print("******* Création d'un modèle ********")
    rf_model = RandomForestClassifier(random_state=42)
    print('******* Entrainement du modèle sur X_train / y_train *******')
    rf_model.fit(X_train, y_train)
   
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
        
    print('******* Optimisation du seuil de classification pour maximiser l\'AUC *******')      
   
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    print('******* Création d\'un dictionnaire pour optimiser les paramètres ******* ')
    
    class_counts = np.bincount(y_train)
    n_classes = len(class_counts)
    class_weights = {i: sum(class_counts)/class_counts[i] for i in range(n_classes)}
    
   
    param_grid = {
    'n_estimators': [10,20,50,75],
    'max_depth': [None,1,5,10,15],
    'class_weight':['balanced',class_weights]   }
    
    print('******* Création d\'un dictionnaire de scoring ******* ')
      
    scoring = {
     'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score),
         'cost': make_scorer(cost_function),
        'precision': make_scorer(precision_score)
    

    }
   
   
    grid_search = GridSearchCV(rf_model, param_grid, scoring=scoring, cv=5,verbose=1, refit='recall')
    print('******* Execution de GridSearchCV sur le modèle ******* ')
   
    grid_search.fit(X_train, y_train)

   
    print("******* Best parameters: ", grid_search.best_params_)
    print("******* Best accuracy: {:.2f}".format(grid_search.best_score_))
    print("******* Best recall: {:.2f}".format(round(grid_search.cv_results_['mean_test_recall'][grid_search.best_index_], 2)))
    print("******* Best cost: {:.2f}".format(round(grid_search.cv_results_['mean_test_cost'][grid_search.best_index_], 2)))
    print("******* Best precision score: {:.2f}".format(round(grid_search.cv_results_['mean_test_precision'][grid_search.best_index_], 2)))

    
   
    print('******* Ajustement du modèle avec les paramètres optimisés ******* ')
    best_rf_model = grid_search.best_estimator_
    best_rf_model.fit(X_train, y_train)

    
    y_pred_best = best_rf_model.predict(X_test)
    y_proba_best = best_rf_model.predict_proba(X_test)[:, 1]
    
    print('******* Optimisation du seil de classification pour maximiser l\'AUC ******* ')      
   
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

   
    y_pred_best = (y_proba_best >= optimal_threshold).astype(int)
    print('******* Scores du modèle après optimisation ******* ')
    
    results = evaluate_classification_model(y_test, y_pred_best, y_proba_best)
   
    results_df = pd.DataFrame(columns=['Modèle', 'Best accuracy', 'Best recall', 'Best cost', 'Best precision Score', 'True Positives', 'True Negatives', 'sampling', 'standardisation', 'Best param', 'feature'])

    true_positives = np.sum((y_test == 1) & (y_pred_best == 1))
    true_negatives = np.sum((y_test == 0) & (y_pred_best == 0))

    results_df.loc[0] = ['Random forest',
                         round(results['accuracy'], 2),
                         round(results['recall'], 2),
                         round(results['cost'], 2),
                         round(results['precision'], 2),
                         true_positives,
                         true_negatives,
                         mode, standardiser, grid_search.best_params_, feature]
   
    print(results_df)
    
    print('******* Affichage de la courbe Roc après optimisation ******* ')
    print(create_roc_auc_plot(best_rf_model,y_pred_best, y_test))
    
    print('******* Affichage du rapport de classification après optimisation ******* ')
    print(create_classification_report(best_rf_model,y_test,y_pred_best))
    
    print('******* Affichage de la matrice de confusion après optimisation ******* ')
    print(create_confusion_matrix_plot(best_rf_model,y_pred_best, y_test))  
    
    return y_pred, y_proba, rf_model, y_pred_best, y_proba_best, best_rf_model, results, results_df

#######################################

def logistic_regression(X_train,y_train,X_test,y_test,x_col,mode='undersampling',standardiser='RobustScaler',feature ='Hypertension'): 
    print('Création d\'un modèle de régression logistique')
    logreg_model = LogisticRegression(random_state=42, max_iter=1000)
    
    print('Entrainement du modèle sur X_train / y_train')
    logreg_model.fit(X_train, y_train)

    y_pred = logreg_model.predict(X_test)
    y_proba = logreg_model.predict_proba(X_test)[:, 1]
        
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
   
    class_counts = np.bincount(y_train)
    n_classes = len(class_counts)
    class_weights = {i: sum(class_counts)/class_counts[i] for i in range(n_classes)}
    
    
    param_grid = {
    'penalty': ['l1','l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'class_weight':['balanced', class_weights]   
    }
    
    print('Création d\'un dictionnaire de scoring')
    
    scoring = {
     'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score),
        'cost': make_scorer(cost_function),
        'precision': make_scorer(precision_score)
    
    }
    
    
    grid_search = GridSearchCV(logreg_model, param_grid, scoring=scoring, cv=5,verbose=1, refit='recall')
    print('Execution de GridSearchCV sur le modèle')
    
    grid_search.fit(X_train, y_train)

    
    print("Best parameters: ", grid_search.best_params_)
    print("Best accuracy: ", grid_search.best_score_)
    print("Best recall: ", grid_search.cv_results_['mean_test_recall'][grid_search.best_index_])
    print("Best cost: ", grid_search.cv_results_['mean_test_cost'][grid_search.best_index_])
    print("Best precision score: ", grid_search.cv_results_['mean_test_precision'][grid_search.best_index_])
    
    
    print('Ajustement du modèle avec les paramètres optimisés')
    best_lr_model = grid_search.best_estimator_
    best_lr_model.fit(X_train, y_train)

    y_pred_best = best_lr_model.predict(X_test)
    y_proba_best = best_lr_model.predict_proba(X_test)[:, 1]
    
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    
    y_pred_best = (y_proba_best >= optimal_threshold).astype(int)
    print('Scores du modèle après optimisation')
    
    results = evaluate_classification_model(y_test, y_pred_best, y_proba_best)
    
    results_df = pd.DataFrame(columns=['Modèle', 'Best accuracy', 'Best recall', 'Best cost', 'Best precision Score', 'True Positives', 'True Negatives', 'sampling', 'standardisation', 'Best param', 'feature'])

    true_positives = np.sum((y_test == 1) & (y_pred_best == 1))
    true_negatives = np.sum((y_test == 0) & (y_pred_best == 0))

    results_df.loc[0] = ['Logistic Regression',
                         round(results['accuracy'], 2),
                         round(results['recall'], 2),
                         round(results['cost'], 2),
                         round(results['precision'], 2),
                         true_positives,
                         true_negatives,
                         mode, standardiser, grid_search.best_params_, feature]
    
    print(results_df)
    
    print('Affichage de la courbe Roc après optimisation')
    print(create_roc_auc_plot(best_lr_model,y_test, y_pred_best))
    
    print('Affichage du rapport de classification après optimisation')
    print(create_classification_report(best_lr_model,y_test,y_pred_best))
    
    print('Affichage de la matrice de confusion après optimisation')
    print(create_confusion_matrix_plot(best_lr_model, y_test, y_pred_best))  
    
    return y_pred, y_proba, logreg_model, y_pred_best, y_proba_best, best_lr_model, results, results_df

#######################################


def svm_classifier(X_train, y_train, X_test, y_test, x_col, mode='undersampling', standardiser='RobustScaler', feature='Hypertension'):
    print("******* Création d'un modèle SVM ********")
    svm_model = SVC(random_state=42, probability=True)
    print('******* Entrainement du modèle sur X_train / y_train *******')
    svm_model.fit(X_train, y_train)
   
    y_pred = svm_model.predict(X_test)
    y_proba = svm_model.predict_proba(X_test)[:, 1]
        
    
    print('******* Création d\'un dictionnaire pour optimiser les paramètres ******* ')
    
    class_counts = np.bincount(y_train)
    n_classes = len(class_counts)
    class_weights = {i: sum(class_counts)/class_counts[i] for i in range(n_classes)}
    
    param_grid = {
        'C': [0.001, 0.01, 0.1],
        'gamma': [0.1, 1, 'auto'],
        'kernel': ['linear', 'rbf', 'poly']
    }
    
    print('******* Création d\'un dictionnaire de scoring ******* ')
      
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score),
        'cost': make_scorer(cost_function),
        'precision': make_scorer(precision_score)
    }
   
    grid_search = GridSearchCV(svm_model, param_grid, scoring=scoring, cv=5, verbose=1, refit='recall')
    print('******* Execution de GridSearchCV sur le modèle ******* ')
   
    grid_search.fit(X_train, y_train)

    print("******* Best parameters: ", grid_search.best_params_)
    print("******* Best accuracy: {:.2f}".format(grid_search.best_score_))
    print("******* Best recall: {:.2f}".format(round(grid_search.cv_results_['mean_test_recall'][grid_search.best_index_], 2)))
    print("******* Best cost: {:.2f}".format(round(grid_search.cv_results_['mean_test_cost'][grid_search.best_index_], 2)))
    print("******* Best precision score: {:.2f}".format(round(grid_search.cv_results_['mean_test_precision'][grid_search.best_index_], 2)))

    print('******* Ajustement du modèle avec les paramètres optimisés ******* ')
    best_svm_model = grid_search.best_estimator_
    best_svm_model.fit(X_train, y_train)

    y_pred_best_svm = best_svm_model.predict(X_test)
    y_proba_best = best_svm_model.predict_proba(X_test)[:, 1]

    print('******* Scores du modèle après optimisation ******* ')
    results = evaluate_classification_model(y_test, y_pred_best_svm, y_proba_best)

    results_df = pd.DataFrame(columns=['Modèle', 'Best accuracy', 'Best recall', 'Best cost', 'Best precision Score', 'True Positives', 'True Negatives', 'sampling', 'standardisation', 'Best param', 'feature'])

    true_positives = np.sum((y_test == 1) & (y_pred_best_svm == 1))
    true_negatives = np.sum((y_test == 0) & (y_pred_best_svm == 0))

    results_df.loc[0] = ['SVM',
                         round(results['accuracy'], 2),
                         round(results['recall'], 2),
                         round(results['cost'], 2),
                         round(results['precision'], 2),
                         true_positives,
                         true_negatives,
                         mode, standardiser, grid_search.best_params_, feature]
    print(results_df)
    
    print('******* Affichage de la courbe Roc après optimisation ******* ')
    print(create_roc_auc_plot(best_svm_model, y_pred_best_svm, y_test))
    
    print('******* Affichage du rapport de classification après optimisation ******* ')
    
    print(create_classification_report(best_svm_model, y_test, y_pred_best_svm))
    
    print('******* Affichage de la matrice de confusion après optimisation ******* ')
    print(create_confusion_matrix_plot(best_svm_model, y_pred_best_svm, y_test))  
    
    return y_pred_best_svm, y_proba_best, svm_model, best_svm_model, results, results_df

#######################################


def cost_function(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost = 10 * fn + fp
    return cost

#######################################

def evaluate_classification_model(y_test, y_pred, y_proba):
       
    accuracy = accuracy_score(y_test, y_pred)
    
    f1 = f1_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred)
    
    recall = recall_score(y_test, y_pred)
    
    auc = roc_auc_score(y_test, y_proba)
    
    cost = cost_function(y_test, y_pred)
    
       
    results = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "cost": cost
       
    }
    
    return results

#######################################

def create_roc_auc_plot(model, y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
    display.plot()
    plt.show()
    
#######################################
    
def create_confusion_matrix_plot(model, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
    disp.plot()
    plt.show()    

#######################################

def create_classification_report(model,y_test,y_pred):
    name = list(set(y_test))
    clsf_report = pd.DataFrame(classification_report(y_test, y_pred,target_names=name, output_dict=True)).round(2).transpose()
    print(clsf_report)
    
#######################################    
    
def feature_importance_rf(model,x_col):
    
    importance_data = pd.DataFrame()
    importance_data['features'] = x_col
    importance_data['importance']= model.feature_importances_
    sns.barplot(data=importance_data[:19], y='features',x='importance')
    plt.show()
    
#######################################

def feature_importance_reg(model, X_train,y_train,x_col):
    model.fit(X_train, y_train)
    indices = np.argsort(model.feature_importances_)[::-1]
    features = []
    for i in range(20):
        features.append(X_train.columns[indices[i]])

    sns.barplot(x=features, y=model.feature_importances_[indices[range(20)]], color=("orange"))
    plt.xlabel('Features importance')
    plt.xticks(rotation=90)
    plt.show()

#######################################

def feature_importance_dummy(model,x_col):
    
    importance_data = pd.DataFrame()
    importance_data['features'] = x_col
    importance_data['coef']= model.coef_
    sns.barplot(data=importance_data[:15], y='features',x='coef')
    plt.show()
    
#######################################

