from ultralytics import YOLO
import os

VERSION = "2025_02_24"
DATASET_PATH = f"/TFG/datasets_labeled/{VERSION}_canicas_dataset/export.yaml"
MODEL_BASE_PATH = f"/TFG/models/canicas/{VERSION}/"

MODELS = [
    f"{VERSION}_canicas_yolo11n.pt",
    # f"{VERSION}_canicas_yolo11s.pt",
    # f"{VERSION}_canicas_yolo11m.pt",
    # f"{VERSION}_canicas_yolo11l.pt",
    # f"{VERSION}_canicas_yolov5nu.pt",
    # f"{VERSION}_canicas_yolov5mu.pt",
    # f"{VERSION}_canicas_yolov8n.pt",
    # f"{VERSION}_canicas_yolov8s.pt",
]

HARDWARE_DEVICES = [
    # 0,
    "dla:0",
    # "dla:1,"
    # "cpu",
]


BATCHES = [
    1,
    # 2,
    # 4,
    # 8,
    # 16
]

# Global PRECISION_CONFIG, will be updated in the loop
PRECISION_CONFIG = {
    "half": False,
    "int8": False,
}

EXPORT_CONFIG = {"image_size": 640, "enable_nms": False, "enable_simplify": True}


def validate_precision_config():
    # This function now uses the global PRECISION_CONFIG
    if PRECISION_CONFIG["half"] and PRECISION_CONFIG["int8"]:
        print(
            "ERROR: Solo se puede elegir un tamaño de precisión (half e int8 no pueden ser True simultáneamente)"
        )
        exit()


def get_precision_suffix():
    # This function now uses the global PRECISION_CONFIG
    if PRECISION_CONFIG["half"]:
        return "FP16"
    elif PRECISION_CONFIG["int8"]:
        return "INT8"
    return "FP32"


def get_hardware_suffix(device, batch_size):
    device_mapping = {0: "GPU", "dla:0": "DLA0", "dla:1": "DLA1"}

    suffix = device_mapping.get(device, "UNKNOWN")

    if batch_size != 1:
        suffix += f"_batch{batch_size}"

    return suffix


def export_model(model_name, device, batch_size=1):
    # This function now uses the global PRECISION_CONFIG
    current_precision_suffix = get_precision_suffix()
    print(
        f"[EXPORT TO TensorRT] Exporting model {model_name} to TensorRT with hardware {device}, batch size {batch_size}, precision {current_precision_suffix}"
    )

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
        nms=EXPORT_CONFIG["enable_nms"],
    )

    hardware_suffix = get_hardware_suffix(device, batch_size)

    source_path = os.path.join(MODEL_BASE_PATH, model_name.replace(".pt", ".engine"))
    target_path = os.path.join(
        MODEL_BASE_PATH,
        f"{model_name.replace('.pt', '')}_{current_precision_suffix}_{hardware_suffix}.engine",
    )

    # Ensure the source file exists before renaming
    if os.path.exists(source_path):
        os.rename(source_path, target_path)
        print(f"Successfully renamed {source_path} to {target_path}")
    else:
        print(f"ERROR: Source file {source_path} not found. Export might have failed.")


def main():
    # Define the precision configurations to iterate over
    # "ninguna" (FP32), "half" (FP16), "int" (INT8)
    PRECISION_ITERATION_OPTIONS = [
        # {"half": False, "int8": False},  # FP32
        {"half": True, "int8": False},  # FP16
        # {"half": False, "int8": True},  # INT8
    ]

    for model_name in MODELS:
        for device in HARDWARE_DEVICES:
            for batch_size in BATCHES:
                for current_precision_setting in PRECISION_ITERATION_OPTIONS:
                    # Update the global PRECISION_CONFIG for the current iteration
                    global PRECISION_CONFIG
                    PRECISION_CONFIG = current_precision_setting

                    # Validate the current precision configuration
                    # This will check if half and int8 are True simultaneously,
                    # which shouldn't happen with PRECISION_ITERATION_OPTIONS.
                    validate_precision_config()

                    export_model(model_name, device, batch_size)


if __name__ == "__main__":
    main()
