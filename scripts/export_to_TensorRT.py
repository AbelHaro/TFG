from ultralytics import YOLO
import os

version = "2024_11_28"
data_dir = f"/TFG/datasets_labeled/2024_11_28_canicas_dataset/data.yaml"
model_path = f"/TFG/models/canicas/{version}/"

models_name = [
    f"{version}_canicas_yolo11n.pt",
    #f"{version}_canicas_yolo11s.pt",
    #f"{version}_canicas_yolo11m.pt",
    #f"{version}_canicas_yolo11l.pt",
    #f"{version}_canicas_yolo11x.pt",
]

hardware = [
    0,
    "dla:0",          
    "dla:1",
]

for model_name in models_name:
    for hw in hardware:
        print(f"[EXPORT TO TensorRT] Exporting model {model_name} to TensorRT with hardware {hw}")   
        
        # Cargar el modelo entrenado
        model = YOLO(model_path + model_name)

        # Exportar el modelo en formato engine
        model.export(
            data=data_dir,
            format="engine",
            half=True,
            device=hw,
            imgsz=640,
        )
        
        # Ajustar el sufijo del nombre del archivo según el hardware
        if hw == 0:
            suffix = "FP16_GPU"
        elif hw == "dla:0":
            suffix = "FP16_DLA0"
        elif hw == "dla:1":
            suffix = "FP16_DLA1"
        else:
            suffix = "UNKNOWN"
        
        # Renombrar el archivo exportado
        src = f"{model_path}{model_name.replace('.pt', '.engine')}"
        dst = f"{model_path}{model_name.replace('.pt', '')}_{suffix}.engine"
        os.system(f"mv {src} {dst}")
