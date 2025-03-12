from sahi import AutoDetectionModel
from sahi.utils.cv import read_image, read_image_as_pil

from ultralytics import YOLO

model_path = "../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n.pt"

image_path = "./prueba.jpg"


detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
)

from sahi.predict import get_sliced_prediction

image_as_pil = read_image_as_pil(image_path)

prediction_result = get_sliced_prediction(
    image=image_as_pil,
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

object_prediction_list = prediction_result.object_prediction_list

from types import SimpleNamespace as Namespace
import torch

def convert_predictions_to_namespace(predictions):
    classes = []
    confidences = []
    boxes_xywh = []

    for pred in predictions:
        # Extraemos valores del bbox
        x_min, y_min, x_max, y_max = pred.bbox.to_xyxy()

        # Calculamos centro y tamaño
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Guardamos los datos
        classes.append(pred.category.id)
        confidences.append(pred.score.value)
        boxes_xywh.append([x_center, y_center, width, height])

    # Convertimos a tensores
    cls_tensor = torch.tensor(classes)
    conf_tensor = torch.tensor(confidences)
    xywh_tensor = torch.tensor(boxes_xywh)

    return Namespace(cls=cls_tensor, conf=conf_tensor, xywh=xywh_tensor)

# predictions es la lista que has mostrado en el ejemplo
resultados = convert_predictions_to_namespace(object_prediction_list)

print(resultados)
print("Clases:", resultados.cls)
print("Confianzas:", resultados.conf)
print("Cajas xywh:", resultados.xywh)








