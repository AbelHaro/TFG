import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Path to the CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
video_files = [
    os.path.join(script_dir, "fps_threads.csv"),
    os.path.join(script_dir, "fps_mp.csv"),
    os.path.join(script_dir, "fps_mh.csv"),
    os.path.join(script_dir, "fps_mpsm.csv"),
]

# Colors for each video - using basic colors
colors = ["blue", "red", "orange", "green"]

# Create a figure with appropriate size
plt.figure(figsize=(12, 6))

# Check if all files exist
all_files_exist = all(os.path.exists(file) for file in video_files)

if all_files_exist:
    max_y = 0
    max_x = 0

    # Plot each video's data
    for i, file in enumerate(video_files):
        print("Loading {}...".format(file))
        data = pd.read_csv(file)
        print("Video {} CSV loaded successfully!".format(i + 1))
        print(data.head())

        # Plot the data
        plt.plot(
            data[data.columns[0]],
            data[data.columns[1]],
            color=colors[i],
            label="Video {}".format(i + 1),
        )

        # Track maximum values
        max_y = max(max_y, data[data.columns[1]].max())
        max_x = max(max_x, data[data.columns[0]].max())

    # Customize the plot
    plt.xlabel("Número de frame")
    plt.ylabel("FPS")
    plt.title("Comparación de FPS en los 4 videos")
    plt.grid(True)
    plt.legend(loc="best")

    # Set axis limits - no padding for x axis
    plt.ylim(0, max_y * 1.1)  # Add 10% padding on top for y axis only
    plt.xlim(0, max_x)  # No padding for x axis
    plt.margins(x=0)  # Remove x margins

    # Save the figure
    save_path = os.path.join(script_dir, "fps_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show the plot
    plt.tight_layout()

else:
    print("Error: One or more CSV files not found!")
    for file in video_files:
        if not os.path.exists(file):
            print("Missing file: {}".format(file))
