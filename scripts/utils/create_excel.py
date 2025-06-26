import csv
import os
import locale


def create_excel(times, total_frames, file="times.csv"):
    output_excels_dir = "../excels/"
    os.makedirs(output_excels_dir, exist_ok=True)
    output_excel_file = os.path.join(output_excels_dir, file)

    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    capture_times = times["capture"]
    tracking_times = times["tracking"]
    processing_times = times["processing"]
    writting_times = times["writting"]
    objects_counts = times["objects_count"]
    frames_per_second_record = times["frames_per_second"]

    print(f"Escribiendo los tiempos en el archivo {output_excel_file}")

    with open(output_excel_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Frame",
                "Captura",
                "Procesamiento",
                "Tracking",
                "Escritura",
                "Cantidad Objetos",
                "Tiempo Total(ms)",
                "Tiempo por Objeto tracking(ms)",
                "FPS",
            ]
        )
        for i in range(total_frames):
            total_frame_time = (
                capture_times[i] + processing_times[i] + tracking_times[i] + writting_times[i]
            )
            row = [
                i,
                format_number(capture_times[i]),
                format_number(processing_times[i]),
                format_number(tracking_times[i]),
                format_number(writting_times[i]),
                objects_counts[i] if i < len(objects_counts) else 0,
                format_number(total_frame_time * 1000),
                format_number(
                    (tracking_times[i] / objects_counts[i]) * 1000 if objects_counts[i] > 0 else 0
                ),
                frames_per_second_record[i] if i < len(frames_per_second_record) else 0,
            ]
            writer.writerow(row)

    print("Terminado")


def format_number(number):
    return f"{number:.6f}".replace('.', ',')
