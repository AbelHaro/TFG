from ultralytics import YOLO
import os

version = "2025_02_24"
data_dir = f"/TFG/datasets_labeled/{version}_canicas_dataset/data.yaml"
model_path = f"/TFG/models/canicas/{version}/"

models_name = [
    f"{version}_canicas_yolo11n.pt",
    #f"{version}_canicas_yolo11s.pt",
    #f"{version}_canicas_yolo11m.pt",
    #f"{version}_canicas_yolo11l.pt",
    # f"{version}_canicas_yolo11x.pt",
]

hardware = [
    0,
    #"dla:0",
    #"dla:1",
]

precision = {
    "half": True,
    "int8": False,
}

batch_size = 8

if precision["half"] and precision["int8"]:
    print("ERROR: Solo se puede elegir un tamaño de precisión")
    exit()

for model_name in models_name:
    for hw in hardware:
        print(f"[EXPORT TO TensorRT] Exporting model {model_name} to TensorRT with hardware {hw}")

        # Cargar el modelo entrenado
        model = YOLO(model_path + model_name)

        # Exportar el modelo en formato engine
        model.export(
            data=data_dir,
            format="engine",
            half=precision["half"],
            int8=precision["int8"],
            device=hw,
            imgsz=640,
            batch=batch_size,
            #simplify=True,
            #nms=True
        )

        # Ajustar el sufijo del nombre del archivo según la precisión
        if precision["half"]:
            precision_suffix = "FP16"
        elif precision["int8"]:
            precision_suffix = "INT8"
        else:
            precision_suffix = "FP32"

        # Ajustar el sufijo del nombre del archivo según el hardware
        if hw == 0:
            hardware_suffix = "GPU"
        elif hw == "dla:0":
            hardware_suffix = "DLA0"
        elif hw == "dla:1":
            hardware_suffix = "DLA1"
        else:
            hardware_suffix = "UNKNOWN"
            
        if batch_size != 1:
            hardware_suffix += f"_batch{batch_size}"

        # Renombrar el archivo exportado
        src = f"{model_path}{model_name.replace('.pt', '.engine')}"
        dst = f"{model_path}{model_name.replace('.pt', '')}_{precision_suffix}_{hardware_suffix}.engine"
        os.system(f"mv {src} {dst}")
