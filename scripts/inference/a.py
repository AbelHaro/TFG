import hardware_stats_usage as hs
import os
from datetime import datetime


objects_count = 40
model_name = "yolo11n"
precision = "FP16"
hardware = "GPU"
mode = f"10W_2CORE"

tegra = "2025-01-17-10-38-14.txt"


output_hardware_stats = f"V2_{model_name}_{precision}_{hardware}_{objects_count}_objects_{mode}.csv"

output_file = output_hardware_stats


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

tegra_stats_output = f"/TFG/excels/tegrastats_outputs/{tegra}"
output_filename = f"/TFG/excels/hardware_stats_usage/{output_file}"

os.makedirs(os.path.dirname(tegra_stats_output), exist_ok=True)
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

hs.create_tegrastats_file(tegra_stats_output, output_filename)