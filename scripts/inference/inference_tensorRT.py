import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Load the TensorRT engine
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Allocate buffers for input and output
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

# Run inference
def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return [out['host'] for out in outputs]

# Preprocess the input image
def preprocess_image(image_path, input_shape=(1, 3, 640, 640)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    return np.expand_dims(image_transposed, axis=0)

# Postprocess the output (adjust for your model's specific output format)
def postprocess_output(output, conf_threshold=0.5):
    boxes, scores, classes = output[0].reshape(-1, 4), output[1], output[2]
    indices = np.where(scores > conf_threshold)
    return boxes[indices], scores[indices], classes[indices]

# Visualize detections
def visualize_detections(image_path, boxes, scores, classes):
    image = cv2.imread(image_path)
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Class: {int(cls)}, Score: {score:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    engine_path = "../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.engine"  # Path to your TensorRT engine file
    image_path = "../datasets_labeled/2024_11_15_canicas_dataset/test/images/383.png"      # Path to the input image

    engine = load_engine(engine_path)
    print("Engine loaded", engine)
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    with engine.create_execution_context() as context:
        input_image = preprocess_image(image_path)
        np.copyto(inputs[0]['host'], input_image.ravel())

        output = do_inference(context, bindings, inputs, outputs, stream)
        boxes, scores, classes = postprocess_output(output)

        visualize_detections(image_path, boxes, scores, classes)
