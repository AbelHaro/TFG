import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde el CSV
df = pd.read_csv("./loss.csv")  # Asegúrate de usar la ruta correcta al archivo CSV

# Crear el gráfico
plt.figure(figsize=(12, 7))

# Definir colores y estilos
sizes = ["n", "s", "m", "l"]
colors = {"n": "blue", "s": "green", "m": "red", "l": "purple"}
linestyles = {"train": "-", "val": "--"}
labels_map = {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large"}


# Iterar sobre los tamaños y graficar
for size_suffix in sizes:
    train_col = f"train_class_loss_{size_suffix}"
    val_col = f"val_class_loss_{size_suffix}"
    color = colors[size_suffix]
    size_label = size_suffix

    if train_col in df.columns:
        plt.plot(
            df["epoca"],
            df[train_col],
            label=f"Train Class Loss ({size_label})",
            color=color,
            linestyle=linestyles["train"],
        )
    if val_col in df.columns:
        plt.plot(
            df["epoca"],
            df[val_col],
            label=f"Validation Class Loss ({size_label})",
            color=color,
            linestyle=linestyles["val"],
        )

# Configurar el gráfico
plt.title("Class Loss durante el entrenamiento y validación por tamaño")
plt.xlabel("Época")
plt.ylabel("Class Loss")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))  # Ajustado para más etiquetas
plt.grid(True)
plt.tight_layout()  # Ajusta el layout para que la leyenda no se corte

plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")  # Guardar la figura

# Mostrar el gráfico
plt.show()
