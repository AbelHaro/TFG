import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Cargar los datos desde el CSV
df = pd.read_csv("./loss.csv")  # Asegúrate de usar la ruta correcta al archivo CSV

# Crear el gráfico
plt.figure(figsize=(15, 8))

# Definir modelos y sus tamaños
model_configs = {
    "yolo11": {"sizes": ["n", "s", "m", "l"], "color": "blue", "marker": "o"},
    "yolov5": {"sizes": ["nu", "mu"], "color": "red", "marker": "s"},
    "yolov8": {"sizes": ["n", "s"], "color": "green", "marker": "^"},
}

linestyles = {"train": "-", "val": "--"}
alpha_values = {"train": 1.0, "val": 0.7}

# Iterar sobre modelos y tamaños
for model, config in model_configs.items():
    base_color = config["color"]
    marker = config["marker"]

    for size in config["sizes"]:
        # Ajustar el color para diferentes tamaños del mismo modelo
        if size in ["n", "nu"]:
            color = base_color
        elif size in ["s", "mu"]:
            color = mcolors.to_rgba(base_color, 0.8)
        elif size == "m":
            color = mcolors.to_rgba(base_color, 0.6)
        else:  # "l"
            color = mcolors.to_rgba(base_color, 0.4)

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

        model_label = {"yolo11": "YOLO11", "yolov5": "YOLOv5", "yolov8": "YOLOv8"}[model]

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
plt.title("Class Loss durante el entrenamiento y validación por modelo y tamaño")
plt.xlabel("Época")
plt.ylabel("Class Loss")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar la figura
plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")

# Mostrar el gráfico
plt.show()
