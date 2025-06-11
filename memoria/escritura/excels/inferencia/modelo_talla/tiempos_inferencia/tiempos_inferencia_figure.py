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
data = pd.read_csv("./tiempos_inferencia.csv")

# Crear el gráfico
plt.figure(figsize=(12, 8))

# Configurar las posiciones de las barras
x = np.arange(len(data["modelo"]))
width = 0.6

# Crear las barras apiladas
p1 = plt.bar(x, data["Tpreprocesado"], width, label=r"Preprocesado", color="#87CEEB")
p2 = plt.bar(
    x,
    data["Tinferencia"],
    width,
    bottom=data["Tpreprocesado"],
    label=r"Inferencia",
    color="#FF6347",
)
p3 = plt.bar(
    x,
    data["Tpostprocesado"],
    width,
    bottom=data["Tpreprocesado"] + data["Tinferencia"],
    label=r"Postprocesado",
    color="#FFD700",
)

# Configurar el gráfico
plt.title(r"Comparaci\'on de tiempos de inferencia por modelo", pad=20)
plt.xlabel(r"Modelo")
plt.ylabel(r"Tiempo (ms)")

# Configurar las etiquetas del eje x
plt.xticks(x, data["modelo"], rotation=45)

# Configurar la leyenda
plt.legend(loc="upper left")

# Configurar grid
plt.grid(True, alpha=0.3, axis="y")

# Configurar límites
plt.ylim(
    0,
    (data["Tpreprocesado"] + data["Tinferencia"] + data["Tpostprocesado"]).max() * 1.1,
)

# Quitar padding de los ejes
plt.margins(0)

# Ajustar márgenes y layout
plt.tight_layout()

# Guardar en alta resolución
plt.savefig("tiempos_inferencia.pdf", bbox_inches="tight")

# Mostrar el gráfico
plt.show()
