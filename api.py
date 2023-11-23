from flask import Flask, render_template, request, redirect, url_for, jsonify
from Data_visualizacion import carga_datos
from modelo import load_model
from modelo import make_prediction


#cargue de los datos para la visualización
data_1, data_2, data_3, data_4, data_5 = carga_datos('data/data.csv')
#lanzamento de la api
app = Flask(__name__)
#se determina e valor para la variable predicción incial
prediccion = '-'

@app.route('/')
def index():
    return render_template('index.html', data1=data_1, data2=data_2, data3=data_3, data5=data_5,predict = prediccion)

#función para cargar los datos del formulario
@app.route('/procesar', methods=['POST'])
def procesar():
    data = request.get_json()
        #se optienen los datos del formulario
    edad = int(data["edad"])
    sintomas = int(data["sintomas"])
    fc_ingreso = int(data["fc_ingreso"])
    fr_ingreso = int(data["fr_ingreso"])
    disnea = int(data["disnea"])
    Sat02 =  int(data["Sat02"])
    oxigeno =  int(data["oxigeno"])
    charlson =  int(data["charlson"])
    news2_calculado =  int(data["news2_calculado"])
    curb65 =  int(data["curb65"])
    hb_ingreso = float(data["hb_ingreso"])
    leucocitos =  int(data["leucocitos"])
    plaquetas =  int(data["plaquetas"])
    creatinina = float(data["creatinina"])
    # En este caso el modelo está en la misma carpeta del notebook
    deployed_model = load_model('./model/modelo1')
    #se crea el listado con las variables en el orden requerido
    new_data = [curb65,disnea,edad,leucocitos,Sat02,news2_calculado,creatinina,sintomas,fr_ingreso,
               fc_ingreso,charlson,oxigeno,hb_ingreso,plaquetas]
    #se ejecuta la predicción
    prediccion = make_prediction(deployed_model, new_data)
    #la predicción da un resultado binario, por lo cual se reformula para que sea en texto
    prediccion = 'Si' if prediccion == [1] else 'No'
    prediccion=f' {prediccion}'
        
    #se recarga la web y se entrega el resultado de la predicción.    
    return jsonify({'result': prediccion})
    

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8001)
    
    
    


  