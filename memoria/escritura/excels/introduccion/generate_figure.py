import matplotlib.pyplot as plt
import pandas as pd

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

# Leer el archivo CSV
data = pd.read_csv("./interes_en_inteligencia_artificial.csv")

# Crear el gráfico
plt.figure(figsize=(10, 8))  # Tamaño adecuado para publicaciones

plt.plot(
    data["Semana"],
    data["Interes"],
    color="blue",
    label=r"Inter\'es en IA",
)

# Títulos y etiquetas con LaTeX
plt.title(r"Inter\'es en Inteligencia Artificial a lo largo de las semanas")
plt.xlabel(r"Semana")
plt.ylabel(r"Pico de inter\'es")

# Escalar ejes x
step = 12
xticks = data["Semana"][::step]
plt.xticks(xticks, rotation=45)

# Leyenda
plt.legend()

# Ajustar disposición
plt.tight_layout()

# Guardar en alta resolución
plt.savefig("interes_en_ia.pdf", bbox_inches="tight")

# Mostrar el gráfico
plt.show()
