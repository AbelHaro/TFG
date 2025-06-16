import argparse
import os
import torch.multiprocessing as mp  # type: ignore

from lib.tcp import tcp_server
from lib.sahi import split_image_with_overlap
import numpy as np
from detection_tracking_pipeline import DEFAULT_SAHI_CONFIG

from unified_pipeline import UnifiedPipeline


def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_objects",
        default="libre",
        type=str,
        choices=["libre", "variable", "0", "18", "40", "48", "60", "70", "88", "176"],
        help="Número de objetos a contar, default=libre",
    )

    parser.add_argument(
        "--model",
        default="yolo11n",
        type=str,
        choices=[
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolov5nu",
            "yolov5mu",
            "yolov8n",
            "yolov8s",
        ],
        help="Talla del modelo, default=yolo11n",
    )

    parser.add_argument(
        "--precision",
        default="FP16",
        type=str,
        choices=["FP32", "FP16", "INT8"],
        help="Precisión del modelo, default=FP16",
    )

    parser.add_argument(
        "--hardware",
        default="GPU",
        type=str,
        choices=["GPU", "DLA0", "DLA1", "ALL", "CPU"],
        help="Hardware a usar, default=GPU",
    )

    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["MAXN", "30W", "15W", "10W"],
        help="Modo de energía a usar",
    )

    parser.add_argument(
        "--tcp", default=False, type=bool, help="Usar conexión TCP, default=False"
    )

    parser.add_argument(
        "--version",
        default="2025_02_24",
        type=str,
        choices=["2025_02_24", "2024_11_28"],
        help="Versión del dataset, default=2025_02_24",
    )

    parser.add_argument(
        "--parallel",
        default="mp_shared_memory",
        type=str,
        choices=["threads", "mp", "mp_shared_memory", "mp_hardware"],
        help="Modo de paralelización a usar, default=threads",
    )

    parser.add_argument(
        "--sahi",
        default=False,
        type=bool,
        help="Usar el modo de procesamiento sahi, default=False",
    )

    parser.add_argument(
        "--max_fps",
        default=None,
        type=int,
        help="FPS máximo que genera la cámara, default=None",
    )

    return parser.parse_args()


def initialize_pipeline(args):
    """Inicializa el pipeline de detección y tracking según el modo de paralelización."""
    mode = f"{args.mode}_{mp.cpu_count()}CORE"
    model_name = args.model

    batch_size = 1

    if args.sahi:
        height = width = 1080
        dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
        _, horizontal_splits, vertical_splits = split_image_with_overlap(
            dummy_image,
            DEFAULT_SAHI_CONFIG["slice_width"],
            DEFAULT_SAHI_CONFIG["slice_height"],
            DEFAULT_SAHI_CONFIG["overlap_pixels"],
        )
        batch_size = horizontal_splits * vertical_splits
        print(
            f"[PROGRAM] Modo sahi activado. División de la imagen: {horizontal_splits}x{vertical_splits}, batch_size de {horizontal_splits*vertical_splits}"
        )

    batch_suffix = f"_batch{batch_size}" if batch_size > 1 else ""

    GPU_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_GPU{batch_suffix}.engine"
    DLA0_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_DLA0{batch_suffix}.engine"
    DLA1_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_DLA1{batch_suffix}.engine"
    CPU_model_path = (
        f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}.pt"
    )

    model_path = (
        GPU_model_path
        if args.hardware == "GPU"
        else (
            DLA0_model_path
            if args.hardware == "DLA0"
            else DLA1_model_path if args.hardware == "DLA1" else CPU_model_path
        )
    )

    if args.num_objects == "libre":
        if args.sahi:
            video_path = f"../../datasets_labeled/videos/test/test_altura_1080x1080.mp4"
        else:
            video_path = (
                f"../../datasets_labeled/videos/contar_objetos_variable_2min.mp4"
            )
    else:
        video_path = (
            f"../../datasets_labeled/videos/contar_objetos_{args.num_objects}_2min.mp4"
        )

    output_dir = "../../inference_predictions/custom_tracker"

    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(
        output_dir,
        f"{args.parallel}_{model_name}_{args.precision}_{args.hardware}_{args.num_objects}_objects_{mode}.mp4",
    )
    sahi_prefix = f"sahi_batch{batch_size}_" if args.sahi else ""
    fps_prefix = f"_{args.max_fps}fps" if args.max_fps else "maxfps"
    output_times = f"{model_name}_{sahi_prefix}{args.precision}_{args.hardware}_{args.num_objects}_objects_{mode}{fps_prefix}"

    print("\n\n[PROGRAM] Opciones seleccionadas:", args, "\n\n")

    if not args.parallel in ["threads", "mp", "mp_shared_memory", "mp_hardware"]:
        raise ValueError(
            "Modo de paralelización no válido. Debe ser 'threads', 'mp', 'mp_shared_memory' o 'mp_hardware'."
        )

    # Crear una instancia del pipeline unificado
    if args.parallel == "mp_hardware":
        pipeline = UnifiedPipeline(
            video_path,
            GPU_model_path,
            output_video_path,
            output_times,
            args.parallel,
            is_tcp=args.tcp,
            sahi=args.sahi,
            max_fps=args.max_fps,
            dla0_model=DLA0_model_path,
            dla1_model=DLA1_model_path,
        )
    else:
        pipeline = UnifiedPipeline(
            video_path,
            model_path,
            output_video_path,
            output_times,
            args.parallel,
            is_tcp=args.tcp,
            sahi=args.sahi,
            max_fps=args.max_fps,
        )

    return pipeline


def main():
    args = parse_arguments()
    detection_tracking_pipeline = initialize_pipeline(args)
    detection_tracking_pipeline.run()


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Fijar método de inicio de multiprocesamiento
    print(f"[PROGRAM] Número de CPU: {mp.cpu_count()}")
    main()
