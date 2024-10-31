from ultralytics import YOLO

# Load your trained model
model = YOLO('best_yolov8_fruit_defect.pt')  # Path to your trained model

# Export the model to TensorRT format
model.export(format="engine", device="0")  # 'device=0' ensures it uses the GPU
