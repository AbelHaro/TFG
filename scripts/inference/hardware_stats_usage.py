import re
import pandas as pd

# Función para procesar una línea de tegrastats
def process_tegrastats_line(line):
    # Extraer el uso y frecuencia de cada núcleo del CPU
    cpu_usage_match = re.findall(r'CPU \[([\d%@, ]+)\]', line)
    cpu_usage = cpu_usage_match[0].split(",") if cpu_usage_match else []
    
    # Crear un diccionario para los núcleos
    cpu_columns = {f"CPU_Core_{i+1}_Usage_%": usage.split("%")[0] for i, usage in enumerate(cpu_usage)}
    cpu_frequencies = {f"CPU_Core_{i+1}_Freq_MHz": usage.split("@")[1] if "@" in usage else None for i, usage in enumerate(cpu_usage)}
    
    # Extraer los consumos energéticos
    power_data = re.findall(r'(GPU|CPU|SOC|CV|VDDRQ|SYS5V) (\d+)mW', line)
    power_dict = {f"{key}_mW": int(value) for key, value in power_data}
    
    # Calcular el consumo energético total
    total_power = sum(power_dict.values())
    power_dict['Total_Power_mW'] = total_power
    
    # Unir toda la información en un solo diccionario
    return {**cpu_columns, **cpu_frequencies, **power_dict}

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
    
    
def create_tegrastats_file(input_file, output_file):
    print(f"[HARDWARE STATS USAGE] Procesando archivo {input_file}...")
    data = parse_tegrastats_file(input_file)
    save_to_csv(data, output_file)
    print(f"[HARDWARE STATS USAGE] Proceso completado")
