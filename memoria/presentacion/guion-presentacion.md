Guión para la presentación del TFG:

# Introducción

Durante los últimos años, la Inteligencia Artificial (IA) ha experimentado un crecimiento en popularidad sin precedentes, transformando nuestra capacidad tecnológica con herramientas revolucionarias. Este avance ha sido impulsado por la disponibilidad de grandes volúmenes de datos, el desarrollo de algoritmos avanzados y las mejoras significativas en el hardware de procesamiento, que han permitido a las máquinas aprender y adaptarse a situaciones complejas.

El progreso en visión por computador ha sido posible gracias a los avances en Redes Neuronales Convolucionales (CNNs), que han revolucionado la capacidad de los sistemas para detectar y clasificar objetos en imágenes y vídeos con una gran precisión y velocidad.

Procesar todos estos datos requiere de un cómputo intensivo, lo que ha llevado a la necesidad de utilizar hardware especializado y la optimización de algoritmos para mejorar el tiempo de procesamiento junto con el consumo energético.

Con todo ello, el objetivo de este trabajo es el desarrollo de un sistema de detección de defectos en objetos en movimiento mediante Redes Neuronales Convolucionales, optimizado para hardware NVIDIA, que permita detectar y clasificar objetos en vídeos en tiempo real.

# Motivación

Los humanos podemos entender el mundo que nos rodea interpretando imágenes y vídeos, algo que no es innato en las máquinas. La visión por computador busca emular esta capacidad humana.

La IA ha revolucionado la tecnología, siendo esencial para soluciones innovadoras. La visión por computador destaca, y dispositivos de bajo consumo como NVIDIA Jetson llevan la IA al *edge computing*, reduciendo latencia y consumo energético, abriendo posibilidades en la industria.

En la industria, la detección y clasificación de objetos en movimiento optimiza procesos, mejora la seguridad y aumenta la eficiencia. La detección manual de defectos es ineficiente y propensa a errores. La automatización con visión artificial reduce costes, aumenta la precisión y mejora la calidad.

# Objetivos

- Estudiar el estado del arte en CNNs, aceleradores y optimizaciones.
- Crear un conjunto de datos para entrenamiento y evaluación.
- Entrenar y validar diferentes modelos CNN para detección de defectos en
tiempo real.
- Implementar un sistema de visión artificial integrado con hardware
NVIDIA.
- Analizar y optimizar cuellos de botella para mejorar rendimiento y
consumo energético.
- Evaluar el sistema con métricas de precisión, latencia y consumo.
- Realizar un análisis comparativo para encontrar la configuración óptima.


# Conceptos Previos - Redes Neuronales Convolucionales

Las Redes Neuronales Convolucionales (CNNs) son un tipo especializado de red neuronal profunda, diseñadas específicamente para procesar datos que tienen una estructura de cuadrícula, como las imágenes. Su arquitectura las hace particularmente efectivas para tareas de visión por computador.

El objetivo principal de una CNN en el contexto de la detección de objetos es doble: primero, localizar la presencia de objetos de interés dentro de una imagen y, segundo, clasificar estos objetos, identificando defectos o características específicas relevantes para la aplicación.

Las CNNs se basan en capas convolucionales para extraer características de las imágenes de manera jerárquica. Estas capas aplican filtros convolucionales que aprenden a detectar patrones visuales en diferentes escalas, desde bordes y texturas hasta formas más complejas.

En la arquitectura de CNNs para detección de objetos, existen dos enfoques principales:

*   **Detectores de dos etapas:** Primero, proponen regiones de interés en la imagen y, luego, clasifican estas regiones. Un ejemplo común es la familia R-CNN (Región-based Convolutional Neural Network).

*   **Detectores de una etapa:** Realizan la detección y clasificación de objetos simultáneamente en una sola pasada por la red, lo que los hace más rápidos. YOLO (You Only Look Once) es un ejemplo popular de este tipo.

# Conceptos Previos - Hardware NVIDIA Jetson

NVIDIA Jetson es una familia de System-on-Chip (SoC) diseñados específicamente para aplicaciones de *edge computing* e IA. Estos dispositivos integran una CPU y una GPU NVIDIA, ofreciendo un equilibrio entre rendimiento y eficiencia energética. Esto los hace ideales para aplicaciones donde el consumo de energía es un factor crítico, como en dispositivos móviles, robótica y sistemas de visión artificial.

El Jetpack SDK de NVIDIA proporciona un conjunto completo de herramientas para desarrollar aplicaciones de IA en los dispositivos Jetson. Incluye bibliotecas, APIs y herramientas de desarrollo que facilitan la implementación y optimización de modelos de *deep learning*. Una de las herramientas más importantes es TensorRT, un optimizador y *runtime* de alto rendimiento para inferencia de *deep learning*. TensorRT permite optimizar los modelos entrenados para que se ejecuten de manera eficiente en la GPU NVIDIA, reduciendo la latencia y aumentando el *throughput*.


# Conceptos Previos - Seguimiento de objetos (MOT)

El seguimiento de objetos es una técnica que permite identificar y seguir la trayectoria de múltiples objetos a lo largo del tiempo en una secuencia de imágenes o un vídeo. A diferencia de la detección de objetos, que solo identifica los objetos en un fotograma individual, el seguimiento de objetos mantiene la identidad de cada objeto a lo largo del tiempo.

El seguimiento de objetos generalmente combina la detección de objetos con el análisis de movimiento. Primero, se detectan los objetos en cada fotograma utilizando un detector de objetos. Luego, se utilizan algoritmos de seguimiento para asociar las detecciones de un fotograma con las detecciones del fotograma anterior, creando así una trayectoria para cada objeto.

BYTETrack es un algoritmo de seguimiento popular que realiza el seguimiento de objetos a partir de las detecciones de un detector de objetos. BYTETrack se destaca por su capacidad para manejar oclusiones y cambios en la apariencia de los objetos, lo que lo hace adecuado para aplicaciones en entornos complejos.

# Propuesta de solución

La propuesta de solución es un sistema de detección de defectos en objetos en movimiento mediante Redes Neuronales Convolucionales, optimizado para hardware NVIDIA. Este sistema como se muestra en la figura, funciona de la siguiente manera:

1. **Entrada de vídeo:** Un sensor, en este caso una cámara, captura un vídeo en tiempo real.

2. **Detección de objetos:** Se utiliza un modelo de detección de objetos para realizar la inferencia del frame actual del vídeo. 

3. **Seguimiento de objetos:** Se aplica un algoritmo de seguimiento de objetos para mantener la identidad de los objetos detectados a lo largo del tiempo.

4. **Escritura de resultados:** Se realizan anotaciones en el frame actual del vídeo, mostrando los objetos detectados y sus trayectorias. También se pueden realizar acciones basadas en los resultados de la detección, como alertas, análisis de datos o activación de actuadores.

Todo el proceso se ejecuta en tiempo real en el edge, sobre un dispositivo NVIDIA Jetson, en este caso, la Jetson AGX Xavier, Jetson AGX Orin o Jetson Orin Nano.

# Desarrollo de la solución - Entrenamiento y validación de modelos

 Creaci´on de un conjunto de datos
con im´agenes de canicas de
distintos colores y defectos.
• Entrenamiento de modelos CNN
para detecci´on de defectos.
• Validaci´on y ajuste de
hiperpar´ametros para mejorar
precisi´on.
• Exportaci´on de los modelos a
TensorRT para optimizar su
rendimiento en hardware NVIDIA