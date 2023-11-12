import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
#from scipy.stats import mannwhitneyu

#Librería para imputación múltiple
import miceforest as mf

#Para modelos

from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

from sklearn import metrics
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score, recall_score,\
precision_recall_fscore_support, accuracy_score, precision_score, confusion_matrix,  f1_score, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# MLFlow para el registro de los experimentos.
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


df = pd.read_excel("data/BD_C19_DSA.xlsx")

síntomas = [ 'Ageusia/disgeusia', 'Cefalea', 'Cianosis', 'Conciencia_alterada', 'Congestion conjuntival',
'Congestion nasal/rinorrea', 'Deshidratacion', 'Diarrea', 'Dias_sintomas_antes_de_consulta',
'Disfonia', 'Disnea', 'Dolor abdominal', 'Dolor lumbar/dorsal/cervical', 'Dolor torácico',
'Erupciones cutaneas', 'Esputo',  'Fatiga', 'Fiebre/escalofrios', 
'Hiporexia', 'Hiposmia/anosmia', 'Mialgias/artralgias', 'Nauseas/emesis', 'Odinofagia',
'Sincope', 'Tos']

Antecedentes = ['Accidente cerebrovascular', 'Cardiopatia isquemica', 
'Cirrosis/enfermedad hepática crónica','Demencia','Diabetes mellitus','Dislipidemia',
'EPOC','Enfermedad de Parkinson','Enfermedad renal crónica','Enfermedad valvular',
'Epilepsia','Falla cardiaca','HPB','Hipertensión arterial','Hipotiroidismo',
'Lupus eritematoso sistémico','Neoplasia hematológica','Neoplasia sólida',
'Obesidad','SAHOS','TVP','Tabaquismo','Taquiarritmias supraventriculares',
'Transplantes','Trastornos psiquiatricos','VIH', 'Asma', 'Artritis reumatoide']

Clinimetria = ['CURB-65_calculado', 'Indice de Charlson', 'Score News2_calculado']

Gravedad = ['Gravedad del COVID-19']

Examen_fisico = ['FC ingreso', 'FR ingreso', 'Hipoventilacion', 'Roncus', 'Sat02 ingreso', 'Sibilancias',
'T ingreso', 'TAM', 'Tirajes','Estertores',]

Farmacos = ['Analgesicos no opioides_si_no', 'Analgesicos opioides_si_no', 'Analgesicos_todos_si_no',
'Antiagregantes plaquetarios_si_no', 'Antibióticos_si_no', 'Anticoagulantes_si_no', 'Antidemenciales_si_no',
'Antidepresivos_si_no', 'Antidiabeticos orales_si_no','Antiepilepticos_si_no','Antihistaminicos_si_no',
'Antiparkinsonianos_si_no','Antipsicoticos_si_no','Antiulcerosos_si_no','Broncodilatadores_si_no',
'Corticoides inhalados_si_no','Corticoides sistémicos_si_no','DMARDS biológicos_si_no','DMARDS sinteticos_si_no',
'Diureticos_si_no','Hipolipemiantes_si_no','Hormona tiroidea_si_no','IECA/ARA2_si_no','Insulinas_si_no',
'Otros antihipertensivos_si_no','Oxigeno suplementario']

General = ['Edad (años)', 'Embarazo_si_no', 'Hombre', 'Lugar de atención', 'Nivel educativo', 'Afiliación SGSSS']

Laboratorios = ['Creatinina ingreso', 'Hb ingreso', 'Leucocitos ingreso', 'Plaquetas ingreso', 
                'Rx_ingreso_Con_alteraciones_si_no', 'Hto ingreso', 'Neutrofilos n absoluto ingreso',
                'Linfocitos n absoluto ingreso']


#eliminamos COVID nosocomial
df2 = df[df['COVID Nosocomial']==0]

#seleccionamos la base dejando por fuera las variables que no sirven o son redundantes
df2 = df2[df2.columns.difference(['Hto ingreso','Neutrofilos n absoluto ingreso',
                                    'Linfocitos n absoluto ingreso','Complicaciones_si_no',
                                    'Complicaciones_n','Posición del paciente en prono',
                                    'DMARDS biológicos_si_no', 'Disnea.1', 'Con_alteraciones_si_no',
                               'Broncodilatadores_corticoides_todos_si_no', 'Antidiabeticos_todos_si_no', 'COVID Nosocomial' ])]

df2['Nivel educativo'] = df2['Nivel educativo'].str.strip()

df2['Nivel educativo'] = df2['Nivel educativo'].map({'primaria':'Primaria', 'Primaria':'Primaria',
                                                             'secundaria':'Secundaria', 'Secundaria':'Secundaria', 
                                                             'Sin informacion':'Sin informacion', 
                                                             'Profesional':'Profesional',
                                                             'Tecnico':'Tecnico', 
                                                             'Postgrado':'Postgrado',})


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2['Afiliación SGSSS'] = le.fit_transform(df2['Afiliación SGSSS'])
df2['Lugar de atención'] = le.fit_transform(df2['Lugar de atención'])
df2['Nivel educativo'] = le.fit_transform(df2['Nivel educativo'])
df2['Gravedad del COVID-19'] = le.fit_transform(df2['Gravedad del COVID-19'])

df3 = df2.copy()

#REALIZAMOS IMPUTACIÓN MÚLTIPLE

# Se crea el kernel. 
kds = mf.ImputationKernel(
  df3,
  save_all_iterations=True,
  random_state=100
)

# Se corre MICE con 5 iteraciones
kds.mice(iterations=5, n_estimators=50)

# Base sin valores perdidos
df_imputed_multiple = kds.complete_data()


y = df_imputed_multiple['Ingreso a UCI']
X = df_imputed_multiple[df_imputed_multiple.columns.difference(['Muerto_si_no','Ingreso a UCI' ])]


#se estandariza la base
scaler = StandardScaler()
data_transformed = scaler.fit_transform(X)
data_transformed = pd.DataFrame(data_transformed, columns = X.columns )

X2 = data_transformed.copy()

seleccionadas = ['CURB-65_calculado', 
'Disnea', 
'Edad (años)', 
'Leucocitos ingreso', 
'Sat02 ingreso',
'Score News2_calculado',
'Creatinina ingreso',
'Dias_sintomas_antes_de_consulta',
'FR ingreso',
'FC ingreso',
'Indice de Charlson',
'Oxigeno suplementario',
'Hb ingreso',
'Plaquetas ingreso']

XFinal=X2[seleccionadas]

# Split 
X_train2, X_test2, y_train2, y_test2 = train_test_split(XFinal, y, test_size=0.20, random_state=123)



# Random Forest

# Se registra el experimento en MLFlow.
experiment = mlflow.set_experiment("RandomForestClassifier")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento.
with mlflow.start_run(experiment_id=experiment.experiment_id):
    
    # Parametros
    n_estimators=500
    criterion='gini'
    max_depth=6
    min_samples_split=2
    min_samples_leaf=1
    min_weight_fraction_leaf=0.0
    max_features='sqrt'
    max_leaf_nodes=None
    min_impurity_decrease=0.0
    bootstrap=True
    oob_score=False
    n_jobs=-1
    verbose=0
    warm_start=False
    class_weight=None
    ccp_alpha=0.0
    max_samples=None
    
    # Estimación
    RF = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, 
                                n_jobs=n_jobs, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                max_samples=max_samples,
                                random_state=123)
    # Entrenamiento
    RF.fit(X_train2, y_train2)
    
    #Evaluación
    pred_RF = RF.predict(X_test2)
    predProba_RF = RF.predict_proba(X_test2)
    
    # Registro de los parámetros
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("criterion", criterion)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("max_samples", max_samples)
    
    # Registro del modelo
    mlflow.sklearn.log_model(RF, "RandomForest_model")
    
    # Cálculo de métricas 
    tn, fp, fn, tp = confusion_matrix(y_test2, pred_RF).ravel()
    RFspec = tn/(tn+fp)
    RFaccuracy = accuracy_score(y_test2, pred_RF)
    RFprecision = precision_score(y_test2, pred_RF)
    RFrecall = recall_score(y_test2, pred_RF)
    RFf1 = f1_score(y_test2, pred_RF)
    # curva ROC
    fpr, tpr, thresholds = roc_curve(y_test2, predProba_RF[:,1], pos_label = 1)
    # Area bajo la Curva - AUC 
    AUCRF = round(auc(fpr, tpr),4)
    
    # Registro de las métricas de interés
    mlflow.log_metric("auc", AUCRF)
    mlflow.log_metric("accuracy", RFaccuracy)
    mlflow.log_metric("precision", RFprecision)
    mlflow.log_metric("sensibilidad", RFrecall)
    mlflow.log_metric("especificidad", RFspec)
    mlflow.log_metric("f1 score", RFf1)
    
    print('\nAccuracy: %.4f' % RFaccuracy)
    print('Precision: %.4f' % RFprecision)
    print('Recall (Sensibilidad): %.4f' % RFrecall)
    print('Especificidad: %.4f' % RFspec)
    print("F1 Score:", round(RFf1,4))
    print('\033[1m'+'AUC: %.4f\n' % AUCRF + '\033[0m')

    Matrix = confusion_matrix(y_test2, pred_RF)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = Matrix, display_labels = [False, True])
    cm_display.plot(cmap=plt.cm.Blues)
    
    # Guardar imagen
    plt.savefig("RF_conf_matrix.png")
    # Registro de la matriz de confusión
    mlflow.log_artifact("RF_conf_matrix.png")


    
    
# XGBoost
    
    
# Se registra el experimento en MLFlow.
experiment = mlflow.set_experiment("XGBClassifier")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento.
with mlflow.start_run(experiment_id=experiment.experiment_id):
    
    # Parametros
    booster='gbtree'
    eta=0.2
    gamma=1
    max_depth=8
    min_child_weight=1
    max_delta_step=0
    subsample=1
    sampling_method='uniform'
    reg_lambda=1
    reg_alpha=0
    tree_method='auto'
    scale_pos_weight=1
    
    
    XGB = XGBClassifier(booster=booster, eta=eta, gamma=gamma, max_depth=max_depth, min_child_weight=min_child_weight,
                        max_delta_step=max_delta_step, subsample=subsample, sampling_method=sampling_method,
                        reg_lambda=reg_lambda, reg_alpha=reg_alpha, tree_method=tree_method, scale_pos_weight=scale_pos_weight)
    
    # Entrenamiento
    XGB.fit(X_train2, y_train2)

    #Evaluación
    pred_XGB = XGB.predict(X_test2)
    predProba_XGB = XGB.predict_proba(X_test2)
    
    # Registro de los parámetros
    mlflow.log_param("learning_rate", eta)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("booster", booster)
    mlflow.log_param("gamma", gamma)
    
    # Registro del modelo
    mlflow.sklearn.log_model(XGB, "XGBoost_model")

    # Cálculo de métricas 
    tn, fp, fn, tp = metrics.confusion_matrix(y_test2, pred_XGB).ravel()
    XGBspec = tn/(tn+fp)
    XGBaccuracy = accuracy_score(y_test2, pred_XGB)
    XGBprecision = precision_score(y_test2, pred_XGB)
    XGBrecall = recall_score(y_test2, pred_XGB)
    XGBf1 = f1_score(y_test2, pred_XGB)
    # curva ROC
    fpr, tpr, thresholds = metrics.roc_curve(y_test2, predProba_XGB[:,1], pos_label = 1)
    # Area bajo la Curva - AUC 
    AUCXGB = round(metrics.auc(fpr, tpr),4)
    
    # Registro de las métricas de interés
    mlflow.log_metric("auc", AUCXGB)
    mlflow.log_metric("accuracy", XGBaccuracy)
    mlflow.log_metric("precision", XGBprecision)
    mlflow.log_metric("sensibilidad", XGBrecall)
    mlflow.log_metric("especificidad", XGBspec)
    mlflow.log_metric("f1 score", XGBf1)
    
    print('\nAccuracy: %.4f' % XGBaccuracy)
    print('Precision: %.4f' % XGBprecision)
    print('Recall (Sensibilidad): %.4f' % XGBrecall)
    print('Especificidad: %.4f' % XGBspec)
    print("F1 Score:", round(XGBf1,4))
    print('\033[1m'+'AUC: %.4f\n' % AUCXGB + '\033[0m')

    Matrix = confusion_matrix(y_test2, pred_XGB)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = Matrix, display_labels = [False, True])
    cm_display.plot(cmap=plt.cm.Blues)
    
    # Guardar imagen
    plt.savefig("XGB_conf_matrix.png")
    # Registro de la matriz de confusión
    mlflow.log_artifact("XGB_conf_matrix.png")



    
# ADABoost


# Se registra el experimento en MLFlow.
experiment = mlflow.set_experiment("AdaBoostClassifier")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento.
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Parametros
    n_estimators = 300
    learning_rate = 0.3
    algorithm = "SAMME"
    random_state = 123

    ADA = AdaBoostClassifier(n_estimators = n_estimators,
                             learning_rate = learning_rate,
                             algorithm=algorithm,
                             random_state=random_state)
    ADA.fit(X_train2, y_train2)


    pred_ADA = ADA.predict(X_test2)
    predProba_ADA = ADA.predict_proba(X_test2)

    # Registro de los parámetros
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("algorithm", algorithm)
    mlflow.log_param("random_state", random_state)

    # Registro del modelo
    mlflow.sklearn.log_model(ADA, "ADABoost_model")

    tn, fp, fn, tp = metrics.confusion_matrix(y_test2, pred_ADA).ravel()
    ADAspec = tn/(tn+fp)
    ADAaccuracy = accuracy_score(y_test2, pred_ADA)
    ADAprecision = precision_score(y_test2, pred_ADA)
    ADArecall = recall_score(y_test2, pred_ADA)
    ADAf1 = f1_score(y_test2, pred_ADA)
    # curva ROC
    fpr, tpr, thresholds = metrics.roc_curve(y_test2, predProba_ADA[:,1], pos_label = 1)
    # Area bajo la Curva - AUC 
    AUCADA = round(metrics.auc(fpr, tpr),4)

    mlflow.log_metric("auc", AUCADA)
    mlflow.log_metric("accuracy", ADAaccuracy)
    mlflow.log_metric("precision", ADAprecision)
    mlflow.log_metric("sensibilidad", ADArecall)
    mlflow.log_metric("especificidad", ADAspec)
    mlflow.log_metric("f1 score", ADAf1)

    print('\nAccuracy: %.4f' % ADAaccuracy)
    print('Precision: %.4f' % ADAprecision)
    print('Recall (Sensibilidad): %.4f' % ADArecall)
    print('Especificidad: %.4f' % ADAspec)
    print("F1 Score:", round(ADAf1,4))
    print('\033[1m'+'AUC: %.4f\n' % AUCADA + '\033[0m')

    Matrix = confusion_matrix(y_test2, pred_ADA)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = Matrix, display_labels = [False, True])
    cm_display.plot(cmap=plt.cm.Blues)
    
    # Guardar imagen
    plt.savefig("ADA_conf_matrix.png")
    # Registro de la matriz de confusión
    mlflow.log_artifact("ADA_conf_matrix.png")










