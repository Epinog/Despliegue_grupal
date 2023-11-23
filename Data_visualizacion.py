#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
#función que carga los datos para las visualizaciones y genera los datos separados para cada gráfico
def carga_datos(ruta):
    #se carga el archivo
    data = pd.read_csv(ruta,sep=';')
    #se generan los datos
    data_1 = data.Edad.value_counts().to_frame().reset_index()
    data_1 = {key:value for key,value in zip(data_1.Edad, data_1['count'])}
    data_2 = {key:value for key,value in zip(data.Edad, data['Indice de Charlson'])}
    data_3 = data.Genero.value_counts().to_frame().reset_index() 
    data_3['count'] = round(data_3['count'] / data_3['count'].sum() * 100,0)
    data_3 = {key:value for key,value in zip(data_3.Genero, data_3['count'])}
    data_4 = data['Afiliación SGSSS'].value_counts().to_frame().reset_index() 
    data_4 = {key:value for key,value in zip(data_4['Afiliación SGSSS'], data_4['count'])}
    data_5 = data['Sat02 ingreso'].value_counts().to_frame().reset_index() 
    data_5 = {float(key.replace(',','.')):value for key,value in zip(data_5['Sat02 ingreso'], data_5['count'])}
    return data_1, data_2, data_3, data_4, data_5

