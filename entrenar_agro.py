import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# 1. Cargar el dataset generado
df = pd.read_csv("historial_cosechas_ica.csv")
print(f"Dataset cargado correctamente. Total de registros: {len(df)}")

# Definir características (X) y variable objetivo (y: toneladas reales)
X = df[['hectareas', 'agua_riego_m3', 'radiacion_solar_horas', 'fertilizante_kg', 'antiguedad_terreno']].values
y = df['toneladas_reales'].values

# 2. Dividir en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalizar los datos (fundamental para redes neuronales y bueno para estabilidad)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el escalador para futuras predicciones en la app
joblib.dump(scaler, "escalador_agro.pkl")
print("Escalador guardado como 'escalador_agro.pkl'.")

# ==========================================
# MODELO 1: Regresión Lineal Simple / Múltiple
# ==========================================
print("\n--- Entrenando Modelo 1: Regresión Lineal ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Regresión Lineal -> RMSE: {rmse_lr:.2f} | R²: {r2_lr:.4f}")

# ==========================================
# MODELO 2: Red Neuronal Profunda (TensorFlow / Keras)
# ==========================================
print("\n--- Entrenando Modelo 2: Red Neuronal Profunda (Keras) ---")
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1) # Capa de salida lineal para predecir toneladas
])

nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar la red neuronal
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=0
)

nn_model.save("modelo_agro.keras")
print("Red neuronal guardada como 'modelo_agro.keras'.")

y_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()

mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"Red Neuronal    -> RMSE: {rmse_nn:.2f} | R²: {r2_nn:.4f}")

# ==========================================
# 4. Generar Evidencia Visual (Matplotlib)
# ==========================================
plt.figure(figsize=(12, 5))

# Gráfico 1: Regresión Lineal (Real vs Predicho)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f"Regresión Lineal (R² = {r2_lr:.2f})")
plt.xlabel("Toneladas Reales")
plt.ylabel("Toneladas Predichas")

# Gráfico 2: Red Neuronal Keras (Real vs Predicho)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_nn, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f"Red Neuronal Keras (R² = {r2_nn:.2f})")
plt.xlabel("Toneladas Reales")
plt.ylabel("Toneladas Predichas")

plt.tight_layout()
plt.savefig("comparacion_modelos_agro.png")
print("\n¡Gráfica comparativa guardada como 'comparacion_modelos_agro.png'!")
plt.show()