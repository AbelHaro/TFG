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
df = pd.read_csv("./loss.csv")  # Asegúrate de usar la ruta correcta al archivo CSV

# Crear el gráfico
plt.figure(figsize=(15, 8))

# Definir colores de la paleta
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

# Definir modelos y sus tamaños
model_configs = {
    "yolo11": {"sizes": ["n", "s", "m", "l"], "marker": "o"},
    "yolov5": {"sizes": ["nu", "mu"], "marker": "s"},
    "yolov8": {"sizes": ["n", "s"], "marker": "^"},
}

linestyles = {"train": "-", "val": "--"}
alpha_values = {"train": 1.0, "val": 0.7}

# Color mapping para cada combinación de modelo y tamaño
color_mapping = {
    ("yolov5", "nu"): colors[0],  # Rojo
    ("yolov5", "mu"): colors[1],  # Azul
    ("yolov8", "n"): colors[2],  # Verde
    ("yolov8", "s"): colors[3],  # Morado
    ("yolo11", "n"): colors[4],  # Naranja
    ("yolo11", "s"): colors[5],  # Amarillo
    ("yolo11", "m"): colors[6],  # Marrón
    ("yolo11", "l"): colors[7],  # Rosa
}

# Iterar sobre modelos y tamaños
for model, config in model_configs.items():
    marker = config["marker"]

    for size in config["sizes"]:
        color = color_mapping[(model, size)]

        # Construir nombres de columnas
        train_col = f"train_class_loss_{model}_{size}"
        val_col = f"val_class_loss_{model}_{size}"

        # Mapear nombres para la leyenda
        size_label = {
            "n": "Nano",
            "s": "Small",
            "m": "Medium",
            "l": "Large",
            "nu": "Nano",
            "mu": "Medium",
        }[size]

        model_label = {"yolo11": "YOLO11", "yolov5": "YOLOv5", "yolov8": "YOLOv8"}[
            model
        ]

        if train_col in df.columns:
            plt.plot(
                df["epoca"],
                df[train_col],
                label=f"{model_label} {size_label} (Train)",
                color=color,
                linestyle=linestyles["train"],
                marker=marker,
                markersize=4,
                markevery=5,
                alpha=alpha_values["train"],
            )
        if val_col in df.columns:
            plt.plot(
                df["epoca"],
                df[val_col],
                label=f"{model_label} {size_label} (Val)",
                color=color,
                linestyle=linestyles["val"],
                marker=marker,
                markersize=4,
                markevery=5,
                alpha=alpha_values["val"],
            )

# Configurar el gráfico
plt.title(r"Class Loss durante el entrenamiento y validaci\'on por modelo y tama\~no")
plt.xlabel(r"\'Epoca")
plt.ylabel(r"Class Loss")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar la figura
plt.savefig("loss_plot.pdf", bbox_inches="tight")

# Mostrar el gráfico
plt.show()
