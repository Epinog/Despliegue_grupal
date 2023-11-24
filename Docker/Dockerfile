FROM python:3.10

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Definir directorio de trabajo
WORKDIR /opt/tablero

# Instalar dependencias
ADD ./dashboard /opt/tablero/
RUN pip install --upgrade pip
RUN pip install -r /opt/tablero/requirements/requirements.txt

# Cambiar propiedad de la carpeta a api-user
RUN chown -R api-user:api-user ./

USER api-user
# Puerto a exponer para la api 
EXPOSE 8001

# Comandos a ejecutar al correr el contenedor
CMD ["python3", "api.py"]
