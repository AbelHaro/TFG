import pycuda.driver as cuda
import time

# Inicializa PyCUDA
cuda.init()

# Obtén el dispositivo (por defecto el primero)
device = cuda.Device(0)

# Crea un contexto en el dispositivo
context = device.make_context()

try:
    while True:
        # Obtén la memoria total y libre de la GPU
        total_memory = device.total_memory()
        free_memory, _ = cuda.mem_get_info()

        # Calcular memoria utilizada
        used_memory = total_memory - free_memory

        # Imprimir los valores en MB
        print(f"Memoria total: {total_memory / (1024 ** 2):.2f} MB")
        print(f"Memoria libre: {free_memory / (1024 ** 2):.2f} MB")
        print(f"Memoria utilizada: {used_memory / (1024 ** 2):.2f} MB")

        # Esperar 3 segundos antes de la siguiente consulta
        time.sleep(3)

finally:
    # Libera el contexto al finalizar
    context.pop()
