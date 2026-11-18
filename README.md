# 🍇 AgroExport Ica - Optimización de Rendimiento de Cosecha

Prototipo tecnológico desarrollado para **Agroexportadora del Sur S.A.C.** enfocado en la estimación inteligente del rendimiento de cosecha (toneladas de uva de mesa por lote) en el valle de Ica, integrando Ciencia de Datos, Deep Learning y una interfaz web en tiempo real.

---

## 🚀 Características Principales
* **Generación de Dataset Agronómico:** Simulación y estructuración de datos históricos adaptados a las condiciones de suelo, clima y riego de Ica.
* **Modelado Predictivo Avanzado:** Entrenamiento y comparación entre **Regresión Lineal** y una **Red Neuronal Profunda (TensorFlow / Keras)**.
* **Backend de Alto Rendimiento:** API REST desarrollada en **FastAPI** para inferencias en tiempo real.
* **Interfaz Web Intuitiva:** Panel interactivo para que los ingenieros agrónomos calculen la producción esperada ingresando parámetros de campo.

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.11+
* **Ciencia de Datos & Machine Learning:** Pandas, NumPy, Scikit-Learn, TensorFlow / Keras, Joblib, Matplotlib
* **Backend:** FastAPI, Uvicorn
* **Frontend:** HTML5, CSS3 (Diseño moderno y adaptable)

---

## 📂 Estructura del Proyecto
```text
AgroExport-Ica/
│
├── generar_dataset.py            # Script para generar el dataset sintético de Ica
├── historial_cosechas_ica.csv    # Dataset agronómico generado
├── entrenar_agro.py              # Script de entrenamiento y evaluación comparativa
├── modelo_agro.keras             # Modelo de Red Neuronal profunda guardado
├── escalador_agro.pkl            # Escalador normalizador (StandardScaler)
├── comparacion_modelos_agro.png  # Gráfica comparativa de desempeño (Real vs Predicho)
├── app.py                        # Backend en FastAPI
├── index.html                    # Interfaz web de usuario
└── README.md                     # Documentación del proyecto
📊 Evaluación de ModelosSe evaluaron dos arquitecturas predictivas (80% entrenamiento, 20% prueba):Regresión Lineal: Ajuste lineal base ($R^2 \approx 1.00$).Red Neuronal Profunda (Keras): Capas densas con activación ReLU, ideal para capturar la no linealidad y el ruido agronómico real ($R^2 = 0.9429$, $\text{RMSE} = 85.18\text{ toneladas}$).⚙️ Instrucciones de Instalación y EjecuciónSigue estos pasos para correr el proyecto localmente en tu máquina:1. Clonar el repositorioBashgit clone [https://github.com/christiangarloz-svg/AgroExprort.git](https://github.com/christiangarloz-svg/AgroExprort.git)
cd AgroExprort
2. Instalar las dependencias necesariasAsegúrate de tener instalado Python y ejecuta en tu terminal:Bashpip install pandas numpy scikit-learn tensorflow fastapi uvicorn matplotlib joblib
3. Generar el dataset (Opcional si ya existe)Bashpython generar_dataset.py
4. Entrenar el modelo y generar métricasBashpython entrenar_agro.py
5. Iniciar la aplicación web (FastAPI)Bashpython -m uvicorn app:app --reload
Una vez iniciado el servidor, abre tu navegador web e ingresa a:
👉 http://127.0.0.1:8000
