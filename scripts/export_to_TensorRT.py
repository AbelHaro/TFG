from ultralytics import YOLO

version = "2024_11_15"
data_dir = f"/TFG/datasets_labeled/2024_11_15_canicas_dataset/data.yaml"
model_path = f"/TFG/models/canicas/{version}/"
output_dir = "../inference_predictions"

models_name = [
    f"{version}_canicas_yolo11n.pt",
    #f"{version}_canicas_yolo11s.pt",
    f"{version}_canicas_yolo11m.pt",
    #f"{version}_canicas_yolo11l.pt",
    #f"{version}_canicas_yolo11x.pt",
]

for model_name in models_name:
    # Cargar el modelo entrenado
    model = YOLO(model_path + model_name)  

    # Exportar el modelo en formato engine
    model.export(
        data=data_dir,
        format="engine",
        half=True,
        #int8=True,
        device=0,
        
    )

    # Cargar el modelo TensorRT exportado
    tensorrt_model = YOLO(model_path + model_name.replace(".pt", ".engine"))

    # Definir la ruta del video
    video_path = "../datasets_labeled/2024_11_15_canicas_dataset/val/images"  # Asegúrate de que esta ruta sea válida

    # Realizar inferencia en el video
    results = tensorrt_model.predict(source=video_path, project=output_dir)  # Asegúrate de que el método predict sea el adecuado
    print(f"Results for {model_name}: {results}")
