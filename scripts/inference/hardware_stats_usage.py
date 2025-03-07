import re
import pandas as pd


# Función para procesar una línea de tegrastats
def process_tegrastats_line(line):
    # Extraer el uso y frecuencia de cada núcleo del CPU
    cpu_usage_match = re.findall(r'CPU \[([\d%@,off ]+)\]', line)
    cpu_usage = cpu_usage_match[0].split(",") if cpu_usage_match else []

    # Crear un diccionario para los núcleos
    cpu_columns = {}
    cpu_frequencies = {}

    for i, usage in enumerate(cpu_usage):
        if usage == "off":
            # Si el núcleo está "off", asignar valores nulos
            cpu_columns[f"CPU_Core_{i+1}_Usage_%"] = 0
            cpu_frequencies[f"CPU_Core_{i+1}_Freq_MHz"] = 0
        else:
            # Extraer el uso y la frecuencia cuando están disponibles
            usage_value, freq_value = usage.split("%") if "%" in usage else (None, None)
            cpu_columns[f"CPU_Core_{i+1}_Usage_%"] = int(usage_value) if usage_value else None
            cpu_frequencies[f"CPU_Core_{i+1}_Freq_MHz"] = (
                int(freq_value[1:]) if freq_value and freq_value.startswith("@") else None
            )

    # Extraer el uso y frecuencia de la GPU
    gpu_match = re.search(r'GR3D_FREQ (\d+)%@\[(\d+)\]', line)
    gpu_data = {
        "GPU_Usage_%": int(gpu_match.group(1)) if gpu_match else 0,
        "GPU_Freq_MHz": int(gpu_match.group(2)) if gpu_match else 0,
    }

    # Extraer los consumos energéticos
    power_data = re.findall(r'(GPU|CPU|SOC|CV|VDDRQ|SYS5V) (\d+)mW', line)
    power_dict = {f"{key}_mW": int(value) for key, value in power_data}

    # Calcular el consumo energético total
    total_power = sum(power_dict.values())
    power_dict['Total_Power_mW'] = total_power

    # Unir toda la información en un solo diccionario
    return {**cpu_columns, **cpu_frequencies, **gpu_data, **power_dict}


# Leer el archivo de tegrastats
def parse_tegrastats_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if "CPU" in line and "mW" in line:  # Filtrar líneas relevantes
                data.append(process_tegrastats_line(line))
    return data


# Guardar el resultado en un archivo CSV
def save_to_csv(data, output_filename):
    df = pd.DataFrame(data)
    df.to_csv(output_filename, index=False)
    print(f"[HARDWARE STATS USAGE] Datos procesados guardados en {output_filename}")


# Función para crear el archivo procesado desde tegrastats
def create_tegrastats_file(input_file, output_file):
    print(f"[HARDWARE STATS USAGE] Procesando archivo {input_file}...")
    data = parse_tegrastats_file(input_file)
    save_to_csv(data, output_file)
    print(f"[HARDWARE STATS USAGE] Proceso completado")
