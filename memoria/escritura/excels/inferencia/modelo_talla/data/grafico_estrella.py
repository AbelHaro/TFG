import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
file_path = "./data_estrella.csv"
df = pd.read_csv(file_path)

# Definir colores bien diferenciados
colors = [
    "#e41a1c",  # Rojo
    "#377eb8",  # Azul
    "#4daf4a",  # Verde
    "#984ea3",  # Morado
    "#ff7f00",  # Naranja
    "#ffff33",  # Amarillo
    "#a65628",  # Marrón
    "#f781bf",  # Rosa
]

# Seleccionar las columnas a graficar con etiquetas descriptivas
labels = [
    "Energía consumida\n(menor es mejor)",
    "Frames/W\n(mayor es mejor)",
    "Tiempo medio inferencia\n(menor es mejor)",
    "mAP50-95\n(mayor es mejor)",
]
metrics = [
    "energia_consumida_J",
    "frames_por_watt",
    "t_inferencia_ms",
    "map50_95",
]
num_vars = len(labels)

# Normalizar las columnas con un rango más suave (0.2 a 1)
df_norm = df.copy()
for col in metrics:
    # Normalización suave que mantiene mejor la proporcionalidad
    min_val = df[col].min()
    max_val = df[col].max()
    df_norm[col] = 0.2 + 0.8 * (df[col] - min_val) / (max_val - min_val)

# Añadir columna con los valores para cerrar el gráfico
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Cerrar el gráfico

# Crear gráfico de radar
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Dibujar cada modelo con su color asignado
for i, row in df_norm.iterrows():
    values = row[metrics].tolist()
    values += values[:1]  # Cerrar el gráfico
    ax.plot(angles, values, label=df["modelo"][i], color=colors[i], linewidth=2)
    ax.fill(angles, values, alpha=0.1, color=colors[i])

# Estética
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# Posicionar la leyenda fuera del gráfico para no solapar con las etiquetas
plt.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
plt.title("Comparativa de Modelos YOLO", size=15, pad=20)
plt.tight_layout()

plt.show()
