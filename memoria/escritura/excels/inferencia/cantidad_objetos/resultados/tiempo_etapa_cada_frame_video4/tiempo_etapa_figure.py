import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Activar estilo LaTeX con fuente académica (idéntica al documento tfgetsinf.cls)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],  # Forzar solo Palatino como en mathpazo
        "text.latex.preamble": r"\usepackage{mathpazo}",  # Mismo paquete que el documento
        "axes.labelsize": 20,  # Igual al texto del documento
        "font.size": 20,  # Tamaño base igual al texto principal
        "legend.fontsize": 18,  # Ligeramente menor que el texto
        "xtick.labelsize": 18,  # Etiquetas del mismo tamaño que texto
        "ytick.labelsize": 18,  # Etiquetas del mismo tamaño que texto
        "axes.titlesize": 22,  # Títulos más grandes
    }
)

# Cargar los datos desde el CSV
data = pd.read_csv("./tiempo_etapa_cada_frame_video4.csv")

# Crear el gráfico
plt.figure(figsize=(12, 8))

# Definir colores para cada etapa
colors = {
    "capture_stage": "#0A3AD6",  # Azul
    "inference_stage": "#EB1616",  # Naranja
    "tracking_stage": "#FFD700",  # Amarillo dorado
    "writing_stage": "#32CD32",  # Verde
}

# Preparar los datos para el gráfico de área apilada
frames = data["frame_number"]
capture = data["capture_stage"]
inference = data["inference_stage"]
tracking = data["tracking_stage"]
writing = data["writing_stage"]

# Crear gráfico de área apilada
plt.fill_between(
    frames,
    0,
    capture,
    color=colors["capture_stage"],
    label=r"capture\_stage",
    alpha=0.8,
)

plt.fill_between(
    frames,
    capture,
    capture + inference,
    color=colors["inference_stage"],
    label=r"inference\_stage",
    alpha=0.8,
)

plt.fill_between(
    frames,
    capture + inference,
    capture + inference + tracking,
    color=colors["tracking_stage"],
    label=r"tracking\_stage",
    alpha=0.8,
)

plt.fill_between(
    frames,
    capture + inference + tracking,
    capture + inference + tracking + writing,
    color=colors["writing_stage"],
    label=r"writing\_stage",
    alpha=0.8,
)

# Configurar el gráfico
plt.title(r"Tiempos de etapa en cada frame del video 4")
plt.xlabel(r"N\'umero de frame")
plt.ylabel(r"Tiempo (s)")

# Configurar la leyenda
plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

# Configurar grid
plt.grid(True, alpha=0.3)

plt.ylim(0, 0.25)

# Quitar padding de los ejes
plt.margins(0)

# Ajustar márgenes y layout
plt.tight_layout()

# Guardar en alta resolución
plt.savefig("tiempo_etapa_video4.pdf", bbox_inches="tight")

# Mostrar el gráfico
plt.show()
