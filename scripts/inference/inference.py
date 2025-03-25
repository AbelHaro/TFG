import argparse
import os
import torch.multiprocessing as mp  # type: ignore

from lib.tcp import tcp_server

# Importación de módulos propios
import detection_tracking_pipeline_with_threads
import detection_tracking_pipeline_with_multiprocesses
import detection_tracking_pipeline_with_multiprocesses_shared_memory
import detection_tracking_pipeline_with_multihardware


def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_objects",
        default="libre",
        type=str,
        choices=["libre", "0", "18", "40", "48", "60", "70", "88", "176"],
        help="Número de objetos a contar, default=libre",
    )

    parser.add_argument(
        "--model_size",
        default="n",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        help="Talla del modelo, default=n",
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
        choices=["GPU", "DLA0", "DLA1", "ALL"],
        help="Hardware a usar, default=GPU",
    )

    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["MAXN", "30W", "15W", "10W"],
        help="Modo de energía a usar",
    )

    parser.add_argument("--tcp", default=False, type=bool, help="Usar conexión TCP, default=False")

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

    return parser.parse_args()


def initialize_pipeline(args):
    """Inicializa el pipeline de detección y tracking según el modo de paralelización."""
    mode = f"{args.mode}_{mp.cpu_count()}CORE"
    model_name = f"yolo11{args.model_size}"

    GPU_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_GPU_batch9.engine"
    DLA0_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_DLA0.engine"
    DLA1_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_DLA1.engine"
    
    
    model_path = GPU_model_path if args.hardware == "GPU" else DLA0_model_path if args.hardware == "DLA0" else DLA1_model_path

    #video_path = f"../../datasets_labeled/videos/contar_objetos_{args.num_objects}_2min.mp4"
    #video_path = f"../../datasets_labeled/videos/test/test_640x640_2400fps.mp4"
    video_path = f"../../datasets_labeled/videos/test/test_1080x1080_2400fps.mp4"
    output_dir = "../../inference_predictions/custom_tracker"

    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(
        output_dir,
        f"{args.parallel}_{model_name}_{args.precision}_{args.hardware}_{args.num_objects}_objects_{mode}.mp4",
    )
    sahi_prefix = "sahi_" if args.sahi else ""
    output_times = f"{model_name}_{sahi_prefix}{args.precision}_{args.hardware}_{args.num_objects}_objects_{mode}"

    print("\n\n[PROGRAM] Opciones seleccionadas:", args, "\n\n")

    pipeline_classes = {
        "threads": detection_tracking_pipeline_with_threads.DetectionTrackingPipelineWithThreads,
        "mp": detection_tracking_pipeline_with_multiprocesses.DetectionTrackingPipelineWithMultiprocesses,
        "mp_shared_memory": (
            detection_tracking_pipeline_with_multiprocesses_shared_memory.DetectionTrackingPipelineWithMultiprocessesSharedMemory
        ),
        "mp_hardware": detection_tracking_pipeline_with_multihardware.DetectionTrackingPipelineWithMultiHardware,
    }

    if args.parallel not in pipeline_classes:
        raise ValueError(
            "Modo de paralelización no válido. Debe ser 'threads', 'mp', 'mp_shared_memory' o 'mp_hardware'."
        )

    return (
        pipeline_classes[args.parallel](
            video_path,
            model_path,
            output_video_path,
            output_times,
            args.parallel,
            args.tcp,
            args.sahi,
        )
        if args.parallel != "mp_hardware"
        else pipeline_classes[args.parallel](
            video_path,
            GPU_model_path,
            DLA0_model_path,
            DLA1_model_path,
            output_video_path,
            output_times,
            args.parallel,
            args.tcp,
            args.sahi,
        )
    )


def main():
    args = parse_arguments()
    detection_tracking_pipeline = initialize_pipeline(args)
    detection_tracking_pipeline.run()


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Fijar método de inicio de multiprocesamiento
    print(f"[PROGRAM] Número de CPU: {mp.cpu_count()}")
    main()
