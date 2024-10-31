import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Ruta del modelo FP32 y donde se guardará el modelo INT8
model_fp32 = '../models/canicas/2024_10_24/2024_10_24_canicas_yolo11n.onnx'
model_int8 = '../models/canicas/2024_10_24/yolov11n_INT8_onnx.onnx'

# Cargar el modelo existente
model = onnx.load(model_fp32)

# Convertir a una versión compatible (IR version 9)
model.ir_version = 9

# Guardar el modelo reconvertido
onnx.save(model, model_fp32)

# Ahora aplicar cuantización dinámica
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QUInt8)

print(f'Modelo cuantizado guardado en: {model_int8}')
