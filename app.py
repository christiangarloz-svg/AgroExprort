from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import os

app = FastAPI(title="AgroExport Ica - Sistema de Optimización de Cosecha")

# Cargar el modelo entrenado y el escalador
modelo = tf.keras.models.load_model("modelo_agro.keras")
escalador = joblib.load("escalador_agro.pkl")

# Estructura de datos recibida desde la web
class LoteInput(BaseModel):
    hectareas: float
    agua_riego_m3: float
    radiacion_solar_horas: float
    fertilizante_kg: float
    antiguedad_terreno: float

@app.get("/", response_class=HTMLResponse)
def home():
    # Carga la interfaz gráfica HTML
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: Archivo index.html no encontrado en la carpeta del proyecto.</h3>"

@app.post("/predecir")
def predecir_cosecha(lote: LoteInput):
    # Organizar los datos en un array para el modelo
    datos_entrada = np.array([[
        lote.hectareas,
        lote.agua_riego_m3,
        lote.radiacion_solar_horas,
        lote.fertilizante_kg,
        lote.antiguedad_terreno
    ]], dtype=float)

    # Normalizar usando el escalador guardado
    datos_escalados = escalador.transform(datos_entrada)
    
    # Realizar la predicción con la Red Neuronal
    prediccion = modelo.predict(datos_escalados, verbose=0)
    toneladas_finales = float(prediccion[0][0])

    return {"toneladas_estimadas": round(toneladas_finales, 2)}