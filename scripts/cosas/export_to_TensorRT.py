from ultralytics import YOLO
import os

VERSION = "2025_02_24"
DATASET_PATH = f"/TFG/datasets_labeled/{VERSION}_canicas_dataset/data.yaml"
MODEL_BASE_PATH = f"/TFG/models/canicas/{VERSION}/"

MODELS = [
    f"{VERSION}_canicas_yolo11n.pt"
]

HARDWARE_DEVICES = [
    0, 
    "dla:0", 
    "dla:1"
    ]

PRECISION_CONFIG = {
    "half": True,
    "int8": False
}

EXPORT_CONFIG = {
    "batch_size": 1,
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

def export_model(model_name, device):
    print(f"[EXPORT TO TensorRT] Exporting model {model_name} to TensorRT with hardware {device}")
    
    model = YOLO(MODEL_BASE_PATH + model_name)
    
    model.export(
        data=DATASET_PATH,
        format="engine",
        half=PRECISION_CONFIG["half"],
        int8=PRECISION_CONFIG["int8"],
        device=device,
        imgsz=EXPORT_CONFIG["image_size"],
        batch=EXPORT_CONFIG["batch_size"],
        simplify=EXPORT_CONFIG["enable_simplify"],
        nms=EXPORT_CONFIG["enable_nms"]
    )
    
    precision_suffix = get_precision_suffix()
    hardware_suffix = get_hardware_suffix(device, EXPORT_CONFIG["batch_size"])
    
    source_path = f"{MODEL_BASE_PATH}{model_name.replace('.pt', '.engine')}"
    target_path = f"{MODEL_BASE_PATH}{model_name.replace('.pt', '')}_{precision_suffix}_{hardware_suffix}.engine"
    
    os.system(f"mv {source_path} {target_path}")

def main():
    validate_precision_config()
    
    for model_name in MODELS:
        for device in HARDWARE_DEVICES:
            export_model(model_name, device)

if __name__ == "__main__":
    main()
