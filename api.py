from flask import Flask, render_template, request, redirect, url_for
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
@app.route('/', methods =["GET", "POST"])
def entrega():    
    if request.method == "POST":
        #se optienen los datos del formulario
        edad = int(request.form.get("edad"))
        sintomas = int(request.form.get("sintomas"))
        fc_ingreso = int(request.form.get("fc_ingreso"))
        fr_ingreso = int(request.form.get("fr_ingreso"))
        disnea = int(request.form.get("disnea"))
        Sat02 = int(request.form.get("Sat02"))
        oxigeno = int(request.form.get("oxigeno"))
        charlson = int(request.form.get("charlson"))
        news2_calculado = int(request.form.get("news2_calculado"))
        curb65 = int(request.form.get("curb65"))
        hb_ingreso = float(request.form.get("hb_ingreso"))
        leucocitos = int(request.form.get("leucocitos"))
        plaquetas = int(request.form.get("plaquetas"))
        creatinina = float(request.form.get("creatinina"))
        # En este caso el modelo está en la misma carpeta del notebook
        deployed_model = load_model('./model/modelo1')
        #se crea el listado con las variables en el orden requerido
        new_data = [curb65,disnea,edad,leucocitos,Sat02,news2_calculado,creatinina,sintomas,fr_ingreso,
                   fc_ingreso,charlson,oxigeno,hb_ingreso,plaquetas]
        #se ejecuta la predicción
        prediccion = make_prediction(deployed_model, new_data)
        #la predicción da un resultado binario, por lo cual se reformula para que sea en texto
        prediccion = 'Si' if prediccion == [1] else 'No'
        
        
    #se recarga la web y se entrega el resultado de la predicción.    
    return render_template("index.html", predict = prediccion, data1=data_1, data2=data_2, data3=data_3, data5=data_5)
    

if __name__ == '__main__':
    app.run(debug=True)
    
    
    


  