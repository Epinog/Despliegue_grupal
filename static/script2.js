
<script>
window.onload = function() {
//Conjunto de datos para la gráfica de la edad
var dps = [
        {% for key,value in data1.items() %}
            { x: {{key}}, y: {{value}} },
        {% endfor %}
];
//Conjunto de datos para la gráfica de dispersión entre la edad y el índice de charlson
var dps2 = [
        {% for key,value in data2.items() %}
            { x: {{key}}, y: {{value}} },
        {% endfor %}
];

//Conjunto de datos para la gráfica de género
var data_dps3 = []
var total = 0


{% for key, value in data3.items() %}
{% if key == 0 %}
    data_dps3.push({ y: {{value}}, label: "Mujer" , color: "#F687B8"});
{% else %}
    data_dps3.push({ y: {{value}}, label: "Hombre" , color: "#1271B3"});
{% endif %}
{% endfor %}

//Conjunto de datos para la gráfica Saturación de Oxígeno
var dps5 = [
        {% for key,value in data5.items() %}
            { x: {{key}}, y: {{value}} },
        {% endfor %}
];

var chart = new CanvasJS.Chart("grafica_1", {
animationEnabled: true,

axisY:{
   
    gridColor: "rgba(1,77,101,.1)",

},
data: [              
{
    // Change type to "doughnut", "line", "splineArea", etc.
    type: "column",
    color: "#014D65",
    dataPoints: dps
}
]
});
chart.render();
//gráfico dispersión edad vs indice de charlson
var chart2 = new CanvasJS.Chart("grafica_2", {
animationEnabled: true,
axisX:{
    title: "Edad",
    gridColor: "rgba(1,77,101,.1)",

},

axisY:{
    title: "Índice de Charlson",
    gridColor: "rgba(1,77,101,.1)",

},
data: [              
{
    // Change type to "doughnut", "line", "splineArea", etc.
    type: "scatter",
    
    dataPoints: dps2
}
]
});
chart2.render();



var chart3 = new CanvasJS.Chart("grafica_3", {
theme: "light2", // "light1", "light2", "dark1", "dark2"
exportEnabled: true,
animationEnabled: true,

data: [{
    type: "pie",
    startAngle: 25,
    toolTipContent: "<b>{label}</b>: {y}%",
    showInLegend: "true",
    legendText: "{label}",
    indexLabelFontSize: 12,
    indexLabel: "{label} - {y}%",
    dataPoints: data_dps3
}]
});
chart3.render();

//grafica 5
var chart5 = new CanvasJS.Chart("grafica_5", {
animationEnabled: true,

axisY:{
    
    gridColor: "rgba(1,77,101,.1)",

},
data: [              
{
    // Change type to "doughnut", "line", "splineArea", etc.
    type: "column",
    color: "#014D65",
    dataPoints: dps5
}
]
});
chart5.render();


};

</script>