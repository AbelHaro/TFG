import tensorrt as trt
import onnx

onnx_model_path = '../models/canicas/2024_10_24/2024_10_24_yolov11n_INT8_onnx.onnx'
trt_model_path = '../models/canicas/2024_10_24/2024_10_24_yolov11n_INT8.trt'

# Cargar el modelo ONNX
onnx_model = onnx.load(onnx_model_path)

# Verificar si el modelo tiene dimensiones de lote explícitas
# (esto es opcional pero puede ayudarte a diagnosticar problemas)
# Puedes usar el módulo onnx para chequear propiedades del modelo aquí

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
config.set_flag(trt.BuilderFlag.INT8)

parser = trt.OnnxParser(network, TRT_LOGGER)

# Parsear el modelo ONNX
if not parser.parse(onnx_model.SerializeToString()):
    for error in range(parser.num_errors):
        print(parser.get_error(error))
    raise Exception("Failed to parse ONNX model")

# Crear el motor
engine = builder.build_engine(network, config)

# Serializar el motor a un archivo
with open(trt_model_path, 'wb') as f:
    f.write(engine.serialize())

print(f'Modelo TensorRT guardado en: {trt_model_path}')
