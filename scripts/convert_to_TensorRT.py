import torch
import torch_tensorrt
from ultralytics import YOLO

# Cargar el modelo YOLO entrenado
model_path = "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n.pt"
yolo = YOLO(model_path)
model = yolo.model.eval().cuda()  # Mover el modelo a GPU y modo evaluación

# Preparar entrada simulada (adaptar tamaño si es necesario)
inputs = [torch.randn((1, 3, 640, 640)).cuda()]  # Dimensión estándar de YOLO

# Compilar el modelo a TensorRT usando Torch-TensorRT
print("Compilando el modelo a TensorRT...")
trt_gm = torch_tensorrt.compile(
    model,
    ir="dynamo",         # Usar PyTorch FX para optimización
    inputs=inputs,       # Definir las dimensiones de entrada
    enabled_precisions={torch.float32},  # Precisión (puedes usar FP16 si es compatible)
    workspace_size=1 << 30  # Tamaño de la memoria intermedia de trabajo (ajusta si es necesario)
)

# Guardar el modelo compilado
output_path = "../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_trt.ep"
torch_tensorrt.save(trt_gm, output_path, inputs=inputs)
print(f"Modelo convertido a TensorRT y guardado en {output_path}")

# Cargar el modelo TensorRT para inferencia
print("Cargando el modelo convertido...")
loaded_model = torch.load(output_path).module()  # Cargar y extraer el módulo del modelo

# Ejecutar inferencia
print("Ejecutando inferencia...")
output = loaded_model(*inputs)
print("Salida del modelo TensorRT:", output)
