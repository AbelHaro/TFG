import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
df = pd.read_csv("./cantidad_objetos_cada_video.csv")

# Crear el gráfico
plt.figure(figsize=(15, 8))

# Definir colores para cada video (igual que en la imagen adjunta)
colors = [
    "#1f77b4",  # Azul - video 1
    "#d62728",  # Rojo - video 2
    "#eecb07",  # Amarillo - video 3
    "#2ca02c",  # Verde - video 4
]

# Configurar videos y sus etiquetas
videos = {
    "video 1": {"color": colors[0], "label": r"video 1"},
    "video 2": {"color": colors[1], "label": r"video 2"},
    "video 3": {"color": colors[2], "label": r"video 3"},
    "video 4": {"color": colors[3], "label": r"video 4"},
}

# Plotear cada video
for video_name, config in videos.items():
    if video_name in df.columns:
        plt.plot(
            df["frame_number"],
            df[video_name],
            color=config["color"],
            label=config["label"],
            linewidth=1.5,
            alpha=0.8,
        )

# Configurar el gráfico con LaTeX
plt.title(r"Cantidad de objetos en cada frame en los v\'ideos de prueba")
plt.xlabel(r"N\'umero de frame")
plt.ylabel(r"Cantidad de objetos")

# Configurar la leyenda
plt.legend(loc="upper right")

# Configurar la cuadrícula
plt.grid(True, alpha=0.3)

# Ajustar márgenes
plt.tight_layout()

# Guardar la figura en alta resolución
plt.savefig("cantidad_objetos_videos.pdf", bbox_inches="tight", dpi=300)

# Mostrar el gráfico
plt.show()
