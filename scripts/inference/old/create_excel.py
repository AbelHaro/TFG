import csv
import os
import locale


def create_excel(times, total_frames, file="times.csv"):
    output_excels_dir = "/TFG/excels/"
    os.makedirs(output_excels_dir, exist_ok=True)
    output_excel_file = os.path.join(output_excels_dir, file)

    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

    capture_times = times["capture"] if "capture" in times else []
    processing_times = times["processing"] if "processing" in times else []
    tracking_times = times["tracking"] if "tracking" in times else []
    writting_times = times["writting"] if "writting" in times else []
    objects_counts = times["objects_count"] if "objects_count" in times else []
    frames_per_second_record = times["frames_per_second"] if "frames_per_second" in times else []
    preprocess_times = times["preprocess"] if "preprocess" in times else []
    inference_times = times["inference"] if "inference" in times else []
    postprocess_times = times["postprocess"] if "postprocess" in times else []

    print(f"Escribiendo los tiempos en el archivo {output_excel_file}")

    with open(output_excel_file, "w", newline="") as csvfile:
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
                "Preprocess",
                "Inference",
                "Postprocess",
            ]
        )
        for i in range(total_frames):
            # total_frame_time = capture_times[i] + processing_times[i] + tracking_times[i] + writting_times[i]
            total_frame_time = 0
            row = [
                i,
                format_number(capture_times[i]) if i < len(capture_times) else 0,
                format_number(processing_times[i] if i < len(processing_times) else 0),
                format_number(tracking_times[i]) if i < len(tracking_times) else 0,
                format_number(writting_times[i]) if i < len(writting_times) else 0,
                objects_counts[i] if i < len(objects_counts) else 0,
                format_number(total_frame_time * 1000),
                # format_number((tracking_times[i] / objects_counts[i]) * 1000 if len(objects_counts) < i and objects_counts[i] > 0 and len(tracking_times) < i else 0),
                0,
                frames_per_second_record[i] if i < len(frames_per_second_record) else 0,
                format_number(preprocess_times[i]) if i < len(preprocess_times) else 0,
                format_number(inference_times[i]) if i < len(inference_times) else 0,
                format_number(postprocess_times[i]) if i < len(postprocess_times) else 0,
            ]
            writer.writerow(row)

    print("Terminado")


def format_number(number):
    return f"{number:.6f}".replace(".", ",")
