import argparse
import detection_tracking_pipeline_with_threads

parser = argparse.ArgumentParser()
parser.add_argument('--num_objects', default=40, type=int, choices=[0, 18, 40, 48, 60, 70, 88, 176], help='Número de objetos a contar, posibles valores: {0, 18, 40, 48, 60, 70, 88, 176}, default=40')
parser.add_argument('--model_size', default='n', type=str, choices=["n", "s", "m", "l", "x"], help='Talla del modelo {n, s, m, l, x}, default=n')
parser.add_argument('--precision', default='FP16', type=str, choices=["FP32", "FP16", "INT8"], help='Precisión del modelo {FP32, FP16, INT8}, default=FP16')
parser.add_argument('--hardware', default='GPU', type=str, choices=["GPU", "DLA0", "DLA1"], help='Hardware a usar {GPU, DLA0, DLA1}, default=GPU')
parser.add_argument('--mode', required=True, default='MAXN', type=str, choices=["MAXN", "30W", "15W", "10W"], help='Modo de energía a usar {MAXN, 30W, 15W, 10W}, default=MAXN')
parser.add_argument('--tcp', default=False, type=bool, help='Usar conexión TCP, default=False')

args = parser.parse_args()

