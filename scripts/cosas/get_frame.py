from ultralytics import YOLO

model_path = "../../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n.pt"

model = YOLO(model_path)

#print(model.model)

print(model)