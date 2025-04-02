from ultralytics import YOLO
import os

VERSION = "2025_02_24"
DATASET_PATH = f"/TFG/datasets_labeled/{VERSION}_canicas_dataset/data.yaml"
MODEL_BASE_PATH = f"/TFG/models/canicas/{VERSION}/"

MODELS = [
    f"{VERSION}_canicas_yolo11n.pt",
    f"{VERSION}_canicas_yolo11s.pt",
    f"{VERSION}_canicas_yolo11m.pt",
    f"{VERSION}_canicas_yolo11l.pt",
]

HARDWARE_DEVICES = [
    0, 
    "dla:0", 
    "dla:1"
]

BATCHES = [
    1, 
    # 2, 
    4, 
    8, 
    # 16
]

PRECISION_CONFIG = {
    "half": True,
    "int8": False,
}

EXPORT_CONFIG = {
    "image_size": 640,
    "enable_nms": False,
    "enable_simplify": True
}

def validate_precision_config():
    if PRECISION_CONFIG["half"] and PRECISION_CONFIG["int8"]:
        print("ERROR: Solo se puede elegir un tamaño de precisión")
        exit()

def get_precision_suffix():
    if PRECISION_CONFIG["half"]:
        return "FP16"
    elif PRECISION_CONFIG["int8"]:
        return "INT8"
    return "FP32"

def get_hardware_suffix(device, batch_size):
    device_mapping = {
        0: "GPU",
        "dla:0": "DLA0",
        "dla:1": "DLA1"
    }
    
    suffix = device_mapping.get(device, "UNKNOWN")
    
    if batch_size != 1:
        suffix += f"_batch{batch_size}"
    
    return suffix

def export_model(model_name, device, batch_size=1):
    print(f"[EXPORT TO TensorRT] Exporting model {model_name} to TensorRT with hardware {device} and batch size {batch_size}")
    
    model = YOLO(os.path.join(MODEL_BASE_PATH, model_name))
    
    model.export(
        data=DATASET_PATH,
        format="engine",
        half=PRECISION_CONFIG["half"],
        int8=PRECISION_CONFIG["int8"],
        device=device,
        imgsz=EXPORT_CONFIG["image_size"],
        batch=batch_size,
        simplify=EXPORT_CONFIG["enable_simplify"],
        nms=EXPORT_CONFIG["enable_nms"]
    )
    
    precision_suffix = get_precision_suffix()
    hardware_suffix = get_hardware_suffix(device, batch_size)
    
    source_path = os.path.join(MODEL_BASE_PATH, model_name.replace('.pt', '.engine'))
    target_path = os.path.join(MODEL_BASE_PATH, f"{model_name.replace('.pt', '')}_{precision_suffix}_{hardware_suffix}.engine")
    
    os.rename(source_path, target_path)

def main():
    validate_precision_config()
    
    for model_name in MODELS:
        for device in HARDWARE_DEVICES:
            for batch_size in BATCHES:
                export_model(model_name, device, batch_size)

if __name__ == "__main__":
    main()