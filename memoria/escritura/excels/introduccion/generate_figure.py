import matplotlib.pyplot as plt
import pandas as pd

# Leer el archivo CSV
data = pd.read_csv('./interes_en_inteligencia_artificial.csv')

# Mostrar las primeras filas para comprobar que se cargó correctamente
print(data.head())

# Crear el gráfico de líneas con puntos
plt.plot(data['Semana'], data['Interes'], color='b', label="Interés en IA")

# Mejorar el gráfico
plt.title("Interés en Inteligencia Artificial a lo largo de las semanas")
plt.xlabel("Semana")
plt.ylabel("Pico de interés")

# Omitir algunos valores de X, por ejemplo, mostrar solo cada 10 valor de x
step = 12  # Omitir valores de X
xticks = data['Semana'][::step]  # Seleccionar cada 'step' valor
plt.xticks(xticks, rotation=45)  # Rotar etiquetas y mostrar solo los valores seleccionados

# Añadir leyenda
plt.legend()

# Mejorar la disposición
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('interes_en_ia.png', dpi=300)  # Guardar la figura


plt.show()
