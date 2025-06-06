"""
Object Detection and Tracking Inference Pipeline

This module provides a comprehensive inference pipeline for object detection and tracking
using YOLO models with support for different hardware configurations, parallelization
modes.

Features:
- Multiple YOLO model variants (YOLOv8, YOLOv11, etc.)
- Hardware acceleration (GPU, DLA0, DLA1, CPU)
- Various precision modes (FP32, FP16, INT8)
- Parallelization strategies (threads, multiprocessing, shared memory)
- Configurable FPS limits
"""

import argparse
import os
import torch.multiprocessing as mp
from unified_pipeline import UnifiedPipeline


def parse_arguments():
    """
    Parse command line arguments for the inference pipeline.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - num_objects: Number of objects to count in video
            - model: YOLO model variant to use
            - precision: Model precision (FP32, FP16, INT8)
            - hardware: Target hardware (GPU, DLA0, DLA1, CPU)
            - mode: Power mode configuration
            - tcp: Whether to use TCP communication
            - version: Dataset version
            - parallel: Parallelization strategy
            - max_fps: Maximum FPS limit
    """
    parser = argparse.ArgumentParser(description="Object detection and tracking inference pipeline")

    # Object counting configuration
    parser.add_argument(
        "--num_objects",
        default="free",
        type=str,
        choices=["free", "variable", "0", "18", "40", "48", "60", "70", "88", "176"],
        help="Number of objects to count in the video, default=free",
    )

    # Model selection
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
        help="YOLO model variant to use, default=yolo11n",
    )

    # Model precision configuration
    parser.add_argument(
        "--precision",
        default="FP16",
        type=str,
        choices=["FP32", "FP16", "INT8"],
        help="Model precision format, default=FP16",
    )

    # Hardware acceleration target
    parser.add_argument(
        "--hardware",
        default="GPU",
        type=str,
        choices=["GPU", "DLA0", "DLA1", "ALL", "CPU"],
        help="Target hardware for inference, default=GPU",
    )

    # Power mode configuration
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["MAXN", "30W", "15W", "10W"],
        help="Power mode configuration (required)",
    )

    # Network communication
    parser.add_argument(
        "--tcp", default=False, type=bool, help="Enable TCP communication, default=False"
    )

    # Dataset version
    parser.add_argument(
        "--version",
        default="2025_02_24",
        type=str,
        choices=["2025_02_24", "2024_11_28"],
        help="Dataset version to use, default=2025_02_24",
    )

    # Parallelization strategy
    parser.add_argument(
        "--parallel",
        default="mp_shared_memory",
        type=str,
        choices=["threads", "mp", "mp_shared_memory", "mp_hardware"],
        help="Parallelization strategy, default=mp_shared_memory",
    )

    # FPS limiting
    parser.add_argument(
        "--max_fps",
        default=None,
        type=int,
        help="Maximum FPS limit for processing, default=None (unlimited)",
    )

    return parser.parse_args()


def initialize_pipeline(args):
    """Initialize the detection and tracking pipeline according to parallelization mode."""
    mode = f"{args.mode}_{mp.cpu_count()}CORE"
    model_name = args.model

    batch_size = 1
    batch_suffix = f"_batch{batch_size}" if batch_size > 1 else ""

    # Define model paths for different hardware configurations
    base_path = (
        f"../../models/canicas/{args.version}/ \
            {args.version}_canicas_{model_name}_{args.precision}"
    )

    GPU_model_path = f"{base_path}_GPU{batch_suffix}.engine"
    DLA0_model_path = f"{base_path}_DLA0{batch_suffix}.engine"
    DLA1_model_path = f"{base_path}_DLA1{batch_suffix}.engine"
    CPU_model_path = f"../../models/canicas/{ \
        args.version}/{args.version}_canicas_{model_name}.pt"

    model_path = (
        GPU_model_path
        if args.hardware == "GPU"
        else (
            DLA0_model_path
            if args.hardware == "DLA0"
            else DLA1_model_path if args.hardware == "DLA1" else CPU_model_path
        )
    )

    # Select video path based on number of objects
    if args.num_objects == "free":
        video_path = "../../datasets_labeled/videos/ \
            contar_objetos_variable_2min.mp4"
    else:
        video_path = f"../../datasets_labeled/videos/ \
            contar_objetos_{args.num_objects}_2min.mp4"

    output_dir = "../../inference_predictions/custom_tracker"

    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(
        output_dir,
        f"{args.parallel}_{model_name}_{args.precision}_{args.hardware}_"
        f"{args.num_objects}_objects_{mode}.mp4",
    )
    fps_prefix = f"_{args.max_fps}fps" if args.max_fps else "maxfps"
    output_times = (
        f"{model_name}_{args.precision}_{args.hardware}_"
        f"{args.num_objects}_objects_{mode}{fps_prefix}"
    )

    print("\n\n[PROGRAM] Selected options:", args, "\n\n")

    if not args.parallel in ["threads", "mp", "mp_shared_memory", "mp_hardware"]:
        raise ValueError(
            "Invalid parallelization mode. Must be 'threads', 'mp', "
            "'mp_shared_memory' or 'mp_hardware'."
        )

    # Create a unified pipeline instance
    if args.parallel == "mp_hardware":
        pipeline = UnifiedPipeline(
            video_path,
            GPU_model_path,
            output_video_path,
            output_times,
            args.parallel,
            is_tcp=args.tcp,
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
            max_fps=args.max_fps,
        )

    return pipeline


def main():
    args = parse_arguments()
    detection_tracking_pipeline = initialize_pipeline(args)
    detection_tracking_pipeline.run()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    print(f"[PROGRAM] Number of CPUs: {mp.cpu_count()}")
    main()
