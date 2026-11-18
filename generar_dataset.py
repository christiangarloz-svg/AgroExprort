import numpy as np
import pandas as pd

# Fijar semilla para que los datos sean reproducibles
np.random.seed(42)

# Número de registros históricos de lotes (ej. 300 campañas/lotes analizados)
n_muestras = 300

# 1. Generar variables independientes con rangos reales para Ica
hectareas = np.random.uniform(2.0, 45.0, n_muestras)
agua_riego_m3 = hectareas * np.random.normal(9500, 800, n_muestras) # Promedio ~9,500 m3 por hectárea
radiacion_solar_horas = np.random.normal(2400, 150, n_muestras)     # Horas de sol acumuladas por campaña
fertilizante_kg = hectareas * np.random.normal(450, 50, n_muestras)  # Kg de abono por hectárea
antiguedad_terreno = np.random.randint(1, 18, n_muestras)            # Años del cultivo

# 2. Simular la variable objetivo (toneladas de uva de mesa) con una fórmula agronómica lógica + ruido aleatorio
# Rendimiento base por hectárea de uva de mesa en Ica (~24-28 toneladas/ha en buen estado)
rendimiento_base = 25.0 

# Factores de influencia en el rendimiento
efecto_agua = 0.0002 * (agua_riego_m3 / hectareas)
efecto_fertilizante = 0.005 * (fertilizante_kg / hectareas)
efecto_sol = 0.001 * (radiacion_solar_horas - 2400)
# Los parrones muy jóvenes (1 año) o muy viejos (>12 años) rinden un poco menos
efecto_antiguedad = -0.15 * (antiguedad_terreno - 6)**2 + 5 

toneladas_por_ha = rendimiento_base + efecto_agua + efecto_fertilizante + efecto_sol + (efecto_antiguedad * 0.1)
# Añadir ruido o imprevistos de campo (plagas menores, clima atípico)
ruido = np.random.normal(0, 3.5, n_muestras)

# Toneladas totales por lote
toneladas_reales = (hectareas * toneladas_por_ha) + ruido
toneladas_reales = np.clip(toneladas_reales, 10, None) # Asegurar valores lógicos positivos

# 3. Crear el DataFrame de Pandas
df_agro = pd.DataFrame({
    'hectareas': np.round(hectareas, 2),
    'agua_riego_m3': np.round(agua_riego_m3, 2),
    'radiacion_solar_horas': np.round(radiacion_solar_horas, 2),
    'fertilizante_kg': np.round(fertilizante_kg, 2),
    'antiguedad_terreno': antiguedad_terreno,
    'toneladas_reales': np.round(toneladas_reales, 2)
})

# 4. Guardar a CSV
nombre_archivo = "historial_cosechas_ica.csv"
df_agro.to_csv(nombre_archivo, index=False)

print(f"¡Dataset '{nombre_archivo}' generado con éxito!")
print(f"Total de registros creados: {n_muestras} lotes agrícolas de Ica.")
print("\nPrimeras 5 filas del dataset:")
print(df_agro.head())