import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

version = "2024_11_28"
model_path = f"/TFG/models/canicas/{version}/"
model_version = "yolo11n.pt"
model_name = f"{version}_canicas_{model_version}"


def prune_model(model, amount=0.1):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, name="weight")
    return model



# Cargar el modelo YOLO entrenado
model_path = "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n.pt"
data_path = "/TFG/datasets_labeled/2024_11_28_canicas_dataset/data.yaml"
output_dir = "../validation_predictions"

model_ultralytics = YOLO(model_path)

results = model_ultralytics.val(data=data_path, batch=16, half=True, plots=True, project=output_dir, conf=0.4, device=0, split='test') 
print("Por defecto mAP50-95: ", results.box.map)

torch_model = model_ultralytics.model

pruned_torch_model = prune_model(torch_model, amount=0.05)




model_ultralytics.model = pruned_torch_model

results = model_ultralytics.val(data=data_path, batch=16, half=True, plots=True, project=output_dir, conf=0.4, device=0, split='test')
print("Pruned mAP50-95: ", results.box.map)

model_ultralytics.save(f"../models/canicas/{version}/{version}_canicas_{model_version.replace('.pt', '')}_pruned.pt")
