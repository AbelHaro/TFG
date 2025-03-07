import argparse
import os
import torch.multiprocessing as mp # type: ignore
import detection_tracking_pipeline_with_threads

parser = argparse.ArgumentParser()
parser.add_argument('--num_objects', default=40, type=int, choices=[0, 18, 40, 48, 60, 70, 88, 176], help='Número de objetos a contar, posibles valores: {0, 18, 40, 48, 60, 70, 88, 176}, default=40')
parser.add_argument('--model_size', default='n', type=str, choices=["n", "s", "m", "l", "x"], help='Talla del modelo {n, s, m, l, x}, default=n')
parser.add_argument('--precision', default='FP16', type=str, choices=["FP32", "FP16", "INT8"], help='Precisión del modelo {FP32, FP16, INT8}, default=FP16')
parser.add_argument('--hardware', default='GPU', type=str, choices=["GPU", "DLA0", "DLA1"], help='Hardware a usar {GPU, DLA0, DLA1}, default=GPU')
parser.add_argument('--mode', required=True, default='MAXN', type=str, choices=["MAXN", "30W", "15W", "10W"], help='Modo de energía a usar {MAXN, 30W, 15W, 10W}, default=MAXN')
parser.add_argument('--tcp', default=False, type=bool, help='Usar conexión TCP, default=False')
parser.add_argument('--version', default="2025_02_24", type=str, choices=["2025_02_24", "2024_11_28"], help='Versión del dataset, default=2025_02_24')
parser.add_argument('--parallel', default="threads", type=str, choices=["threads", "processes", "processes_shared_memory"], help='Modo de paralelización a usar {threads, processes, processes_shared_memory}, default=threads')

args = parser.parse_args()

def main():
    num_objects = args.num_objects
    model_size = args.model_size
    precision = args.precision
    hardware = args.hardware
    mode = mode = f"{args.mode}_{mp.multiprocessing.cpu_count()}CORE"
    tcp = args.tcp
    version = args.version
    parallel_mode = args.parallel

    print("\n\n[PROGRAM] Opciones seleccionadas: ", args, "\n\n")

    model_name = f"yolo11{model_size}"

    model_path = f"../../../models/canicas/{version}/{version}_canicas_{model_name}_{precision}_{hardware}.engine"
    video_path = f'../../../datasets_labeled/videos/contar_objetos_{num_objects}_2min.mp4'
    output_dir = '../../../inference_predictions/custom_tracker'

    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{parallel_mode}_{model_name}_{precision}_{hardware}_{num_objects}_objects_{mode}.mp4")
    output_times = f"{model_name}_{precision}_{hardware}_{num_objects}_objects_{mode}"
        
    detection_tracking_pipeline = detection_tracking_pipeline_with_threads.DetectionTrackingPipelineWithThreads(video_path, model_path, output_video_path,output_times, parallel_mode, tcp, args.tcp)
    
    detection_tracking_pipeline.run()
    
if __name__ == '__main__':
    mp.multiprocessing.set_start_method('spawn')   
    print("[PROGRAM] Number of cpu : ", mp.multiprocessing.cpu_count())
    main()