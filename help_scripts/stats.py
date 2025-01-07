import ast

def read_statistics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Leer estadísticas reales
    real_counts = {}
    for line in lines:
        if ":" in line and not line.startswith("{") and not line.startswith("TOTAL"):
            cls, count = line.split(":")
            real_counts[cls.strip()] = int(count.strip())

    # Leer estadísticas predichas (última línea con diccionario)
    predicted_counts = ast.literal_eval(lines[-1].strip())

    return real_counts, predicted_counts

def calculate_statistics(real_counts, predicted_counts):
    total_real = sum(real_counts.values())
    total_predicted = sum(predicted_counts.values())

    # Calcular precisión por clase
    precision_by_class = {}
    for cls in real_counts:
        real = real_counts[cls]
        pred = predicted_counts.get(cls, 0)
        precision = min(real, pred) / real if real > 0 else 0
        precision_by_class[cls] = precision

    # Calcular precisión total
    total_precision = sum(min(real_counts[cls], predicted_counts.get(cls, 0)) for cls in real_counts) / total_real

    return precision_by_class, total_precision

def main():
    file_path = "../inference_predictions/custom_tracker/count.yaml"
    real_counts, predicted_counts = read_statistics(file_path)

    precision_by_class, total_precision = calculate_statistics(real_counts, predicted_counts)

    print("\nEstadísticas de Precisión:")
    for cls, precision in precision_by_class.items():
        print(f"Clase '{cls}': Precisión: {precision:.2%}")

    print(f"\nPrecisión Total: {total_precision:.2%}")

if __name__ == "__main__":
    main()
