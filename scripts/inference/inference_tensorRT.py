import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Inicializa automáticamente PyCUDA

# Ruta al motor TensorRT generado por trtexec
ENGINE_PATH = "../../models/canicas/2024_11_28/out.engine"

# Configuración de logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    """Carga el motor TensorRT desde un archivo."""
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """Prepara los buffers de entrada y salida del motor."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Asignación de memoria
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Diferencia entre entrada y salida
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
        bindings.append(int(device_mem))

    return inputs, outputs, bindings, stream

def do_inference(engine, inputs, outputs, bindings, stream):
    """Realiza la inferencia en TensorRT."""
    context = engine.create_execution_context()

    # Copiar datos a la memoria del dispositivo
    [cuda.memcpy_htod_async(inp["device"], inp["host"], stream) for inp in inputs]

    # Ejecutar inferencia
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copiar resultados de vuelta a la memoria del host
    [cuda.memcpy_dtoh_async(out["host"], out["device"], stream) for out in outputs]

    # Sincronizar el stream
    stream.synchronize()

    # Retornar resultados
    return [out["host"] for out in outputs]

def preprocess_image(image_path, input_shape):
    """Preprocesar una imagen para adaptarla al modelo."""
    from PIL import Image
    image = Image.open(image_path).resize(input_shape)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalizar
    image = np.transpose(image, (2, 0, 1))  # Convertir a formato CHW
    return np.expand_dims(image, axis=0).astype(np.float32)

def postprocess_results(results):
    """Postprocesar resultados del modelo."""
    # Implementa la lógica específica para interpretar la salida del modelo.
    return results

# Ruta a una imagen de entrada
IMAGE_PATH = "../../datasets_labeled/2024_11_28_canicas_dataset/test/images/16.png"

# Tamaño de entrada del modelo
INPUT_SHAPE = (300, 300)  # Según lo especificado en los argumentos de trtexec

# Cargar el motor
engine = load_engine(ENGINE_PATH)

# Preparar buffers
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Preprocesar la imagen
input_data = preprocess_image(IMAGE_PATH, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
np.copyto(inputs[0]["host"], input_data.ravel())

# Ejecutar inferencia
results = do_inference(engine, inputs, outputs, bindings, stream)

# Postprocesar resultados
final_results = postprocess_results(results)
print("Resultados de inferencia:", final_results)
