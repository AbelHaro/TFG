from ultralytics import YOLO
import os
import cv2
from ultralytics.trackers.bot_sort import BOTSORT  # type: ignore
from argparse import Namespace
import torch


def xywh_to_tlwh(xywh):
    if isinstance(xywh, torch.Tensor):
        tlwh = xywh.clone()
    elif isinstance(xywh, np.ndarray):
        tlwh = xywh.copy()
    else:
        tlwh = list(xywh)  # Para listas

    tlwh[0] = xywh[0] - xywh[2] / 2  # t = x - w/2
    tlwh[1] = xywh[1] - xywh[3] / 2  # l = y - h/2
    return tlwh


model_path = "../../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n.pt"
video_path = "../../datasets_labeled/2025_02_24_canicas_dataset/train/images/133.png"

# Verificar la existencia de los archivos
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El modelo '{model_path}' no existe.")

if not os.path.exists(video_path):
    raise FileNotFoundError(f"El video '{video_path}' no existe.")

# Cargar el modelo
model = YOLO(model_path, task="detect")

# Configuración del modelo
model(device="cuda:0", conf=0.5, half=True, imgsz=(640, 640), augment=True)

# Cargar la imagen
frame = cv2.imread(video_path)

# Configuración de los parámetros de tracking
args = Namespace(
    proximity_thresh=0.5,
    appeareance_thresh=0.5,
    track_buffer=30,
    appearance_thresh=0.5,
    gmc_method="sparseOptFlow",
    with_reid=True,
)

bot_sort = BOTSORT(args, frame_rate=30)

# Preprocesamiento, inferencia y postprocesamiento
preprocessed = model.predictor.preprocess([frame])
output = model.predictor.inference(preprocessed)
results = model.predictor.postprocess(output, preprocessed, [frame])

# Mostrar los resultados del modelo
print(f"[DEBUG] El modelo ha detectado: {len(results[0].boxes.xywh.cpu())} objetos.")

print("[DEBUG] Resultados:", results[0].boxes)

xywh = results[0].boxes.xywh.cpu()
conf = results[0].boxes.conf.cpu()
cls = results[0].boxes.cls.cpu()

print("[DEBUG] xywh:", xywh)
tlwh = xywh_to_tlwh(xywh)
print("[DEBUG] tlwh:", tlwh)

print("[DEBUG] conf:", conf)
print("[DEBUG] cls:", cls)


# Crear un índice ascendente para cada objeto detectado
xywh_with_idx = torch.cat(
    (xywh, torch.arange(len(xywh)).view(-1, 1).float()), dim=1
)  # Agrega el índice

print("[DEBUG] xywh_with_idx:", xywh_with_idx)

# Realizar el tracking
tracks = bot_sort.init_track(dets=xywh_with_idx, scores=conf, cls=cls, img=frame)

# Mostrar los tracks obtenidos
for track in tracks:
    print("[DEBUG] Track:", track)

# bot_sort.multi_predict(tracks)
