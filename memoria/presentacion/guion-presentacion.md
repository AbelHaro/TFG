# Guión para la presentación del TFG

# Introducción

En los últimos años, la Inteligencia Artificial ha revolucionado la tecnología con herramientas que han transformado por completo nuestra capacidad de innovación. Este avance se debe a tres factores clave: la enorme cantidad de datos disponibles, el desarrollo de modelos más sofisticados y las mejoras en el hardware especializado en deep learning.

Dentro de la IA, la visión por computador ha experimentado un progreso extraordinario gracias a las Redes Neuronales Convolucionales. Estos modelos permiten a los sistemas detectar y clasificar objetos en imágenes y vídeos con una gran precisión y velocidad.

Sin embargo, procesar tal volumen de datos exige una gran capacidad de cómputo. Por ello, es fundamental optimizar los modelos y utilizar hardware especializado para reducir los tiempos de procesamiento y el consumo energético.


# Motivación

En la industria, la inspección manual de defectos es un proceso lento y propenso a errores. Para optimizar estos procesos, se busca emular la capacidad humana de interpretar el mundo a través de la vista, pero en las máquinas.

La visión por computador permite replicar esta habilidad, y gracias a dispositivos de bajo consumo como la familia NVIDIA Jetson, podemos llevar la IA al "edge". Esto reduce la latencia y el consumo, abriendo un mundo de posibilidades para la automatización de la detección de defectos, aumentando la seguridad y mejorando la eficiencia.

# Objetivos

- **Objetivo Principal:** Desarrollar un sistema de detección de defectos en objetos en movimiento utilizando CNNs, optimizado para hardware NVIDIA.

- **Objetivos Específicos:**
  
- **Estudiar el estado del arte en CNNs y aceleradores hardware:**  
  Investigar las últimas arquitecturas de Redes Neuronales Convolucionales (CNNs) y los aceleradores de hardware disponibles, especialmente aquellos diseñados para plataformas NVIDIA, para comprender las técnicas más avanzadas en detección de objetos y optimización del rendimiento.

- **Crear un conjunto de datos para entrenamiento y evaluación:**  
  Desarrollar un conjunto de datos diverso y representativo que contenga imágenes de objetos con y sin defectos, que permita entrenar modelos de detección robustos y evaluar su rendimiento de manera precisa.

- **Entrenar y validar diferentes modelos CNN para detección de defectos en tiempo real:**  
  Entrenar varios modelos de CNN utilizando el conjunto de datos creado, ajustando los hiperparámetros y validando su rendimiento para lograr una detección de defectos precisa y eficiente en tiempo real.

- **Implementar un sistema de visión artificial:**  
  Desarrollar un sistema completo de visión artificial que combine la detección de objetos basada en CNNs con el procesamiento en hardware NVIDIA, aprovechando las capacidades de aceleración y optimización de estas plataformas.

- **Analizar y optimizar cuellos de botella para mejorar el rendimiento y el consumo energético:**  
  Identificar y optimizar los cuellos de botella en el sistema, tanto a nivel de software como de hardware, para mejorar el rendimiento general y reducir el consumo energético, permitiendo una operación más eficiente en entornos con recursos limitados.

- **Evaluar el sistema con métricas de precisión, latencia y consumo:**  
  Evaluar exhaustivamente el sistema utilizando métricas clave como la precisión en la detección de defectos, la latencia en el procesamiento de imágenes y el consumo energético, para comprender su rendimiento en diferentes condiciones y configuraciones.

- **Realizar un análisis comparativo para encontrar la configuración óptima:**  
  Comparar diferentes configuraciones del sistema, incluyendo diferentes modelos de CNN, parámetros de hardware y estrategias de optimización, para identificar la configuración que ofrece el mejor equilibrio entre precisión, rendimiento y eficiencia energética.


# Conceptos Previos - Redes Neuronales Convolucionales

"Las Redes Neuronales Convolucionales (CNN) son un tipo de red neuronal diseñado específicamente para procesar datos con estructura matricial, como las imágenes.

En la detección de objetos, una CNN tiene dos tareas principales: localizar objetos de interés en una imagen y clasificarlos, identificando si tienen defectos o no. Para ello, utilizan capas convolucionales que aprenden a reconocer patrones visuales, desde bordes y texturas hasta formas complejas.

Existen dos enfoques principales en la arquitectura de estas redes:

- **Detectores de dos etapas (Two-Stage):** Como la familia R-CNN, primero proponen regiones donde podría haber un objeto y luego las clasifican.
- **Detectores de una etapa (One-Stage):** Como YOLO, realizan la detección y clasificación en un solo paso, lo que los hace mucho más rápidos.

# Conceptos Previos - Hardware NVIDIA Jetson

NVIDIA Jetson es una familia de System-on-Chip (SoC) diseñados específicamente para aplicaciones de *edge computing* e IA. Estos dispositivos integran una CPU y una GPU NVIDIA, ofreciendo un equilibrio entre rendimiento y eficiencia energética. Esto los hace ideales para aplicaciones donde el consumo de energía es un factor crítico, como en dispositivos móviles, robótica y sistemas de visión artificial.

El Jetpack SDK de NVIDIA proporciona un conjunto completo de herramientas para desarrollar aplicaciones de IA en los dispositivos Jetson. Incluye bibliotecas, APIs y herramientas de desarrollo que facilitan la implementación y optimización de modelos de *deep learning*. Una de las herramientas más importantes es TensorRT, un optimizador y *runtime* de alto rendimiento para inferencia de *deep learning*. TensorRT permite optimizar los modelos entrenados para que se ejecuten de manera eficiente en la GPU NVIDIA, reduciendo la latencia y aumentando el *throughput*.


# Conceptos Previos - Seguimiento de objetos (MOT)

El seguimiento de objetos es una técnica que busca identificar y seguir la trayectoria de múltiples objetos a lo largo del tiempo en una secuencia de imágenes o un vídeo. A diferencia de la detección de objetos, que solo identifica los objetos en un fotograma individual, el seguimiento de objetos mantiene la identidad de cada objeto a lo largo del tiempo.

Para ello, combina la detección de objetos con algoritmos que asocian las detecciones de un fotograma con las del siguiente. En este proyecto he utilizado BYTETrack, un algoritmo muy eficaz que funciona bien incluso cuando los objetos se tapan entre sí.

# Propuesta de solución

La solución que he desarrollado es un sistema de detección de defectos en tiempo real, optimizado para hardware NVIDIA. El sistema sigue cuatro pasos:

1.  **Entrada de vídeo:** Una cámara captura el vídeo en tiempo real.
2.  **Detección de objetos:** Un modelo de CNN analiza cada fotograma para detectar objetos.
3.  **Seguimiento de objetos:** El algoritmo BYTETrack realiza el seguimiento de los objetos detectados para mantener su identidad.
4.  **Escritura de resultados:** El sistema anota el vídeo con las detecciones y trayectorias, y podría activar otras acciones, como alertas o actuadores.

Todo el proceso se ejecuta en tiempo real en el edge, sobre un dispositivo NVIDIA Jetson, en este caso, la Jetson AGX Xavier, Jetson AGX Orin o Jetson Orin Nano.

# Desarrollo de la solución - Entrenamiento y validación de modelos

Para generalizar la detección de defectos, se eligieron canicas como objeto de estudio. Las canicas presentan varias ventajas: su forma simple facilita el análisis, ofrecen una gran variedad de imperfecciones posibles, no caducan como la fruta, son económicas, fáciles de conseguir en grandes cantidades y su tamaño es uniforme.

El primer paso del desarrollo fue crear un conjunto de datos con imágenes de canicas, tanto con defectos como sin ellos.

Con este dataset, entrené varios modelos de CNN para que aprendieran a detectar estos defectos. Después, ajusté sus hiperparámetros para mejorar la precisión y, finalmente, los exporté a formato TensorRT para optimizar su rendimiento en el hardware de NVIDIA.

# Desarrollo de la solución - Segmentación de las etapas

El sistema propuesto se divide en cuatro etapas principales: captura de vídeo, detección de objetos, seguimiento de objetos y escritura de resultados.

- El objetivo de segmentar estas etapas es permitir que cada una opere de forma independiente, mejorando así la velocidad y la eficiencia del sistema.
- Se han planteado cuatro enfoques de segmentación para optimizar el procesamiento secuencial:
  1. Segmentación por hilos
  2. Segmentación por procesos
  3. Segmentación por procesos con memoria compartida
  4. Segmentación heterogénea

## Desarrollo de la solución - Segmentación por procesos con memoria compartida

Este enfoque mejora el anterior al reducir la latencia:

- Cada etapa se ejecuta en un proceso separado, pero todos comparten memoria.
- ↑ Permite procesamiento concurrente con menor latencia de comunicación.
- Las colas de comunicación se implementan sobre memoria compartida, evitando la sobrecarga de los *pipes*.


# Desarrollo de la solución - Prueba de concepto

## Desarrollo de la solución - Construcción de una cinta transportadora para una prueba de concepto

Para validar el sistema en un entorno real, se construyó una cinta transportadora como prueba de concepto.

- Se utilizó un motor para mover la cinta, una Raspberry Pi Pico para controlar el motor y un servo para rechazar los objetos defectuosos.
- El objetivo principal fue demostrar la capacidad del sistema para detectar objetos en movimiento y clasificarlos en tiempo real, simulando un proceso de inspección industrial automatizado.
# Resultados

## Evaluación del rendimiento del sistema

Para evaluar el rendimiento del sistema, existen diferentes variables de configuración a considerar:

- La cantidad de objetos en el vídeo.
- El tipo de segmentación utilizado.
- El modelo de CNN y su tamaño.
- La precisión numérica del modelo (FP32, FP16 o INT8).
- El modo de energía del dispositivo Jetson.
- Y el modelo de Jetson utilizado.

Además, se han considerado 2 formas de procesar los vídeos:

- **Ejecución a máxima capacidad:** El sistema procesa todo el vídeo sin descartar frames.
- **Procesamiento en tiempo real:** Los fotogramas se suministran al sistema a 30 FPS, descartando aquellos que no se pueden procesar a tiempo.

## Resultados - Cantidad de objetos

Se ha evaluado el rendimiento del sistema frente a diferentes cantidades de objetos presentes en el vídeo. Para ello, se han utilizado cuatro vídeos de prueba con los siguientes niveles de carga:

1. Media de 17 objetos
2. Media de 43 objetos
3. Media de 84 objetos
4. Carga variable entre 0 y 180 objetos

Para el resto de parámetros del sistema, se han empleado las configuraciones más óptimas determinadas previamente en la sección de análisis comparativo.

A partir del análisis de los resultados y aplicando un modelo de regresión, se ha determinado que el sistema es capaz de mantener una tasa de procesamiento en tiempo real de **30 FPS** hasta un máximo aproximado de **100 objetos simultáneos**. Superado este umbral, el rendimiento comienza a degradarse progresivamente, afectando tanto a la latencia como a la precisión de detección.


## Resultados - Tipo de segmentación

También se ha evaluado el rendimiento del sistema bajo diferentes estrategias de segmentación, utilizando el vídeo con una media de 84 objetos. Al igual que en la evaluación anterior, el resto de parámetros del sistema han sido fijados en sus valores óptimos según el análisis comparativo previo.

Entre todas las opciones analizadas, la **segmentación por procesos con memoria compartida** ha demostrado ser la más eficiente. Esta configuración ha logrado:

- Un **speedup de 2.47x** respecto a la versión secuencial.
- Un **menor consumo energético**, optimizando el uso de los recursos del dispositivo Jetson.

Esto confirma que una segmentación adecuada tiene un impacto directo en la capacidad del sistema para operar en tiempo real y con eficiencia energética.

# Conclusiones - Cumplimiento de objetivos

Los objetivos planteados en este Trabajo de Fin de Grado se han alcanzado con éxito, cumpliendo tanto los aspectos técnicos como experimentales previstos. A continuación, se detallan los principales hitos logrados:

## Objetivos alcanzados

- **Estudio del estado del arte:**  
  Se realizó un análisis exhaustivo de redes neuronales convolucionales (CNNs), aceleradores hardware y plataformas NVIDIA Jetson. Se abordaron con éxito los desafíos de compatibilidad y configuración.

- **Creación del conjunto de datos:**  
  Se generó un conjunto de datos y vídeos de entrenamiento que simulan condiciones reales de operación, con objetos de distintos colores y defectos bajo una variabilidad controlada.

- **Entrenamiento de modelos CNN:**  
  Se entrenaron múltiples modelos de las familias YOLOv5, YOLOv8 y YOLOv11 en diferentes precisiones (FP32, FP16, INT8), realizando un ajuste exhaustivo de hiperparámetros para maximizar la precisión y eficiencia.

- **Implementación del sistema integrado:**  
  Se desarrolló un sistema de visión artificial que integra detección en tiempo real con seguimiento de objetos mediante BYTETrack, manteniendo la identidad de cada objeto a lo largo del tiempo.

- **Análisis y optimización de cuellos de botella:**  
  Se evaluaron distintas estrategias de segmentación del sistema (por hilos, por procesos, con memoria compartida y heterogénea), identificando la segmentación por procesos con memoria compartida como la opción más eficiente.

- **Evaluación con métricas completas:**  
  Se realizaron experimentos exhaustivos midiendo precisión, latencia y consumo energético bajo diferentes configuraciones de hardware Jetson, obteniendo una visión completa del rendimiento del sistema.


# Correciones de la presentación
- Motivacion : Quitar los 2 primeros puntos y cambiar la primera imagen
-Quitar de la introducción el objetivo general
- pagina 7 añdir imagenes de las plataformas y
pagina 8 reducir texto
pagina 9 ampliar la imagen para que se vea mejor
pagina 10, añadir los diferentes hiperparámetros usados y entrenamientos hechos
pagina 11 menos texto, primer punto repartir las etapas
Segmentacion de laa etapas exlicar solo por procesos con memoria compartida, imagenes iguales
- pagina 16 quitar la palabra simple, mejorar el texto
- pagnia 18 añadir el numero de casos, mejorar la presentación y  la escritura
pagina 20 tabla mas grande, en negrita la mejor, poner solo parametros optimos
video con un monton de canicas a la vez
