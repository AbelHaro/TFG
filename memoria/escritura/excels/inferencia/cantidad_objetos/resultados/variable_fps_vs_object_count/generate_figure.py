import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

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

# Path to the CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
object_count_file = os.path.join(script_dir, "object_count.csv")
fps_file = os.path.join(script_dir, "fps.csv")

# Read the CSV files
if os.path.exists(object_count_file) and os.path.exists(fps_file):
    print(f"File {object_count_file} found!")
    object_count_data = pd.read_csv(object_count_file)
    print("Object count CSV loaded successfully!")
    print(object_count_data.head())

    fps_data = pd.read_csv(fps_file)
    print(f"File {fps_file} found!")
    print("FPS CSV loaded successfully!")

    # Create a figure with appropriate size and two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot FPS on the primary axis (switched)
    fps_color = "tab:blue"
    ax1.set_xlabel(r"N\'umero de frame")
    ax1.set_ylabel(r"FPS", color=fps_color)  # Modified: Set y-axis label color
    ax1.plot(
        fps_data[fps_data.columns[0]],
        fps_data[fps_data.columns[1]],
        color=fps_color,
        label=fps_data.columns[1],
    )  # Modified: Added label for legend
    ax1.tick_params(axis="y", labelcolor=fps_color)

    # Plot object count on the secondary axis (switched)
    objects_color = "tab:red"
    ax2.set_ylabel(
        r"Cantidad de objetos", color=objects_color
    )  # Modified: Set y-axis label color
    ax2.plot(
        object_count_data[object_count_data.columns[0]],
        object_count_data[object_count_data.columns[1]],
        color=objects_color,
        label=object_count_data.columns[1],
    )  # Modified: Added label for legend
    ax2.tick_params(axis="y", labelcolor=objects_color)

    # Set x-axis and y-axis limits
    x_max = object_count_data[object_count_data.columns[0]].max()
    plt.xlim(0, x_max)

    # Set y-axis limits starting from 0
    y1_max = fps_data[fps_data.columns[1]].max()
    y2_max = object_count_data[object_count_data.columns[1]].max()
    ax1.set_ylim(0, y1_max * 1.1)  # Add 10% padding on top only
    ax2.set_ylim(0, y2_max * 1.1)  # Add 10% padding on top only

    # Set x-axis tick label size
    ax1.tick_params(axis="x")

    # Add title and grid
    plt.title(
        r"FPS y Cantidad de Objetos por frame en el video 4 (carga variable)",
        pad=20,
    )
    ax1.grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, loc="best"
    )  # Modified: Use labels from get_legend_handles_labels

    # Apply tight layout before saving
    plt.tight_layout()  # Modified: Moved before savefig

    # Save the figure
    save_path = os.path.join(script_dir, "fps_vs_object_count.pdf")
    plt.savefig(save_path, bbox_inches="tight")

    # Show the plot
    plt.show()

else:
    print(f"Error: File {object_count_file} or {fps_file} not found!")
