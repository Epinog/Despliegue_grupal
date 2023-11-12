from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)



@app.route('/')
def index():
    data = pd.read_csv('data/data.csv', encoding='ANSI', sep=';')
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

    return render_template('index.html', data1=data_1, data2=data_2, data3=data_3, data5=data_5)



if __name__ == '__main__':
    app.run(debug=True)
    
# <!DOCTYPE html>
# <html>
# <head>
#     <title>To-Do List</title>
# </head>
# <body>
#     <h1>Lista de Tareas</h1>
#     <ul>
#         {% for task in tasks %}
#         <li>{{ task }}</li>
#         {% endfor %}
#     </ul>
#     <form action="/add_task" method="post">
#         <label for="task">Nueva tarea:</label>
#         <input type="text" id="task" name="task">
#         <button type="submit">Agregar tarea</button>
#     </form>
# </body>
# </html>
