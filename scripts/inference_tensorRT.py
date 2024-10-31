import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver
import tensorrt as trt

# Configuración de TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    """Carga el modelo TensorRT desde un archivo .engine."""
    with open(engine_file_path, "rb") as f:
        return trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """Asigna memoria para los buffers de entrada y salida."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        buffer = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
        bindings.append(int(buffer))

        if engine.binding_is_input(binding):
            inputs.append(buffer)
        else:
            outputs.append(buffer)

    return inputs, outputs, bindings, stream

def infer(engine, inputs, outputs, bindings, stream, image):
    """Realiza la inferencia utilizando el modelo TensorRT."""
    # Copiar la imagen a la memoria de entrada
    cuda.memcpy_htod(inputs[0], image)

    # Ejecutar la inferencia
    engine.execute_v2(bindings=bindings)

    # Copiar el resultado de la memoria de salida
    cuda.memcpy_dtoh(outputs[0], outputs[0])

    return outputs[0]

def preprocess_image(image_path, input_shape):
    """Carga y preprocesa la imagen para la inferencia."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[2], input_shape[1]))  # Resize to (width, height)
    image = image.transpose((2, 0, 1))  # Cambia el orden a (C, H, W)
    image = np.ascontiguousarray(image, dtype=np.float32)  # Asegura que sea contiguo
    image = image[np.newaxis, :]  # Añade la dimensión del batch
    return image

def main():
    # Configuración del modelo y archivo de imagen
    engine_file_path = '../models/canicas/2024_10_24/2024_10_24_canicas_yolo11n_INT8.engine'  # Cambia por tu archivo .engine
    image_path = '../datasets_labeled/2024_10_24_canicas_dataset/val/images/94.png'  # Cambia por la ruta de tu imagen

    # Cargar el motor TensorRT
    engine = load_engine(engine_file_path)

    # Asignar buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Preprocesar la imagen
    input_shape = (1, 3, 640, 640)  # Cambia según las dimensiones de entrada de tu modelo
    image = preprocess_image(image_path, input_shape)

    # Realizar la inferencia
    output = infer(engine, inputs, outputs, bindings, stream, image)

    # Procesar la salida (por ejemplo, imprimir el resultado)
    print("Resultado de la inferencia:", output)

if __name__ == '__main__':
    main()
