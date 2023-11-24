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
    # Convert input data to DMatrix format
    input_dmatrix = xgb.DMatrix(np.array(input_data).reshape(1, -1))

    # Make prediction
    prediction = model.predict(input_dmatrix)

    # Assuming binary classification, you might want to convert the output to a class (0 or 1)
    # If it's a multiclass classification, you might need a different approach
    predicted_class = (prediction > 0.5).astype(int)

    return predicted_class


