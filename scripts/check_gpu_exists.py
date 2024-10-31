import torch


def exists_gpu():
    # Obtener el número de GPUs disponibles
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 0:
        print(f"Número de GPUs disponibles: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No se encontraron GPUs disponibles.")
        return False
    
    return True