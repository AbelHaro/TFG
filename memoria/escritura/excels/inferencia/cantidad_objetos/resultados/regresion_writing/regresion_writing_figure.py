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
data = pd.read_csv("./data.csv")

# Crear el gráfico
plt.figure(figsize=(12, 8))

# Extraer datos
x = data["objects_count"]
y = data["writing_stage"]

# Crear scatter plot
plt.scatter(x, y, color="#32CD32", alpha=0.8, s=30, label=r"writing\_stage")

# Usar ecuación específica: 2.11E-03x^0.56
coeff = 2.11e-03
exponent = 0.56

# Crear línea de regresión usando la ecuación específica
x_line = np.linspace(x.min(), x.max(), 100)
y_line = coeff * (x_line**exponent)

# Dibujar línea de regresión
plt.plot(
    x_line,
    y_line,
    color="#FF6347",
    linewidth=2,
    label=f"$2.11E-03x^{{0.56}}$, $R^2 = 0.87$",
)

# Configurar el gráfico
plt.title(
    r"Tiempo de la etapa de escritura en funci\'on del n\'umero de objetos detectados",
    pad=20,
)
plt.xlabel(r"N\'umero de objetos detectados")
plt.ylabel(r"Tiempo de escritura (s)")

# Configurar la leyenda
plt.legend(loc="upper left")

# Configurar grid
plt.grid(True, alpha=0.3)

# Configurar límites
plt.xlim(0, x.max() * 1.05)
plt.ylim(0, y.max() * 1.05)

# Quitar padding de los ejes
plt.margins(0)

# Ajustar márgenes y layout
plt.tight_layout()

# Guardar en alta resolución
plt.savefig("regresion_writing.pdf", bbox_inches="tight")

# Mostrar el gráfico
plt.show()
