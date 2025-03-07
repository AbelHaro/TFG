import argparse
import os
import torch.multiprocessing as mp  # type: ignore

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
        default=40,
        type=int,
        choices=[0, 18, 40, 48, 60, 70, 88, 176],
        help="Número de objetos a contar, default=40",
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
        choices=["GPU", "DLA0", "DLA1, ALL"],
        help="Hardware a usar, default=GPU",
    )

    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["MAXN", "30W", "15W", "10W"],
        help="Modo de energía a usar",
    )

    parser.add_argument("--tcp", action="store_true", help="Usar conexión TCP")

    parser.add_argument(
        "--version",
        default="2025_02_24",
        type=str,
        choices=["2025_02_24", "2024_11_28"],
        help="Versión del dataset, default=2025_02_24",
    )

    parser.add_argument(
        "--parallel",
        default="threads",
        type=str,
        choices=["threads", "mp", "mp_shared_memory", "mp_hardware"],
        help="Modo de paralelización a usar, default=threads",
    )

    return parser.parse_args()


def initialize_pipeline(args):
    """Inicializa el pipeline de detección y tracking según el modo de paralelización."""
    mode = f"{args.mode}_{mp.cpu_count()}CORE"
    model_name = f"yolo11{args.model_size}"

    model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_{args.hardware}.engine"

    GPU_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_GPU.engine"
    DLA0_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_DLA0.engine"
    DLA1_model_path = f"../../models/canicas/{args.version}/{args.version}_canicas_{model_name}_{args.precision}_DLA1.engine"

    video_path = (
        f"../../datasets_labeled/videos/contar_objetos_{args.num_objects}_2min.mp4"
    )
    output_dir = "../../inference_predictions/custom_tracker"

    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(
        output_dir,
        f"{args.parallel}_{model_name}_{args.precision}_{args.hardware}_{args.num_objects}_objects_{mode}.mp4",
    )
    output_times = f"{model_name}_{args.precision}_{args.hardware}_{args.num_objects}_objects_{mode}"

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
            "Modo de paralelización no válido. Debe ser 'threads', 'mp' o 'mp_shared_memory'."
        )

    return (
        pipeline_classes[args.parallel](
            video_path,
            model_path,
            output_video_path,
            output_times,
            args.parallel,
            args.tcp,
            args.tcp,
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
            args.tcp,
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
