#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xgboost as xgb

#función para cargar el modelo
def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model


#función para hacer la predicción
def make_prediction(model, input_data):
    # Convertir los datos a  formato DMatrix
    input_dmatrix = xgb.DMatrix(np.array(input_data).reshape(1, -1))

    # Realizar la predicción
    prediction = model.predict(input_dmatrix)

    # Convertir el desenlace en 1 o 0
    predicted_class = (prediction > 0.5).astype(int)

    return predicted_class


