# Guión para la presentación del TFG

# Introducción

Durante los últimos años, la Inteligencia Artificial ha experimentado un crecimiento sin precedentes en popularidad, transformando nuestra capacidad tecnológica mediante herramientas revolucionarias. Este avance ha sido impulsado por la disponibilidad de grandes volúmenes de datos, el desarrollo de algoritmos avanzados y las mejoras significativas en el hardware de procesamiento, que han permitido a las máquinas aprender y adaptarse a situaciones complejas.

El progreso en visión por computador ha sido posible gracias a los avances en Redes Neuronales Convolucionales (CNNs), que han revolucionado la capacidad de los sistemas para detectar y clasificar objetos en imágenes y vídeos con gran precisión y velocidad.

El procesamiento de todos estos datos requiere un cómputo intensivo, lo que ha llevado a la necesidad de utilizar hardware especializado y optimizar algoritmos para mejorar tanto el tiempo de procesamiento como el consumo energético.

En este contexto, el objetivo de este trabajo es el desarrollo de un sistema de detección de defectos en objetos en movimiento mediante Redes Neuronales Convolucionales, optimizado para hardware NVIDIA, que permita detectar y clasificar objetos en vídeos en tiempo real.

# Motivación

Los humanos podemos entender el mundo que nos rodea interpretando imágenes y vídeos, una capacidad que no es innata en las máquinas. La visión por computador busca emular esta capacidad humana.

La IA ha revolucionado la tecnología, siendo esencial para el desarrollo de soluciones innovadoras. En este ámbito destaca la visión por computador, mientras que dispositivos de bajo consumo como NVIDIA Jetson permiten llevar la IA al *edge computing*, reduciendo la latencia y el consumo energético, lo que abre nuevas posibilidades en la industria.

En el entorno industrial, la detección y clasificación de objetos en movimiento optimiza procesos, mejora la seguridad y aumenta la eficiencia. La detección manual de defectos resulta ineficiente y propensa a errores humanos. La automatización mediante visión artificial permite reducir costes, aumentar la precisión y mejorar la calidad global del proceso.

# Objetivos
- **Estudiar el estado del arte en CNNs y aceleradores hardware:**  
  Investigar las últimas arquitecturas de Redes Neuronales Convolucionales (CNNs) y los aceleradores de hardware disponibles, especialmente aquellos diseñados para plataformas NVIDIA, para comprender las técnicas más avanzadas en detección de objetos y optimización del rendimiento.

- **Crear un conjunto de datos para entrenamiento y evaluación:**  
  Desarrollar un conjunto de datos diverso y representativo que contenga imágenes de objetos con y sin defectos, que permita entrenar modelos de detección robustos y evaluar su rendimiento de manera precisa.

- **Entrenar y validar diferentes modelos CNN para detección de defectos en tiempo real:**  
  Entrenar varios modelos de CNN utilizando el conjunto de datos creado, ajustando los hiperparámetros y validando su rendimiento para lograr una detección de defectos precisa y eficiente en tiempo real.

- **Implementar un sistema de visión artificial integrado con hardware NVIDIA:**  
  Desarrollar un sistema completo de visión artificial que combine la detección de objetos basada en CNNs con el procesamiento en hardware NVIDIA, aprovechando las capacidades de aceleración y optimización de estas plataformas.

- **Analizar y optimizar cuellos de botella para mejorar el rendimiento y el consumo energético:**  
  Identificar y optimizar los cuellos de botella en el sistema, tanto a nivel de software como de hardware, para mejorar el rendimiento general y reducir el consumo energético, permitiendo una operación más eficiente en entornos con recursos limitados.

- **Evaluar el sistema con métricas de precisión, latencia y consumo:**  
  Evaluar exhaustivamente el sistema utilizando métricas clave como la precisión en la detección de defectos, la latencia en el procesamiento de imágenes y el consumo energético, para comprender su rendimiento en diferentes condiciones y configuraciones.

- **Realizar un análisis comparativo para encontrar la configuración óptima:**  
  Comparar diferentes configuraciones del sistema, incluyendo diferentes modelos de CNN, parámetros de hardware y estrategias de optimización, para identificar la configuración que ofrece el mejor equilibrio entre precisión, rendimiento y eficiencia energética.


# Conceptos Previos - Redes Neuronales Convolucionales

Las Redes Neuronales Convolucionales (CNNs) son un tipo especializado de red neuronal profunda, diseñadas específicamente para procesar datos que tienen una estructura matricial, como las imágenes. Su arquitectura las hace particularmente efectivas para tareas de visión por computador.

El objetivo principal de una CNN en el contexto de la detección de objetos es doble: primero, localizar la presencia de objetos de interés dentro de una imagen y, segundo, clasificar estos objetos, identificando defectos o características específicas relevantes para la aplicación.

Las CNNs se basan en capas convolucionales para extraer características de las imágenes de manera jerárquica. Estas capas aplican filtros convolucionales que aprenden a detectar patrones visuales en diferentes escalas, desde bordes y texturas hasta formas más complejas.

En la arquitectura de CNNs para detección de objetos, existen dos enfoques principales:

- **Detectores de dos etapas:** Primero, proponen regiones de interés en la imagen y, luego, clasifican estas regiones. Un ejemplo común es la familia R-CNN (Región-based Convolutional Neural Network).

- **Detectores de una etapa:** Realizan la detección y clasificación de objetos simultáneamente en una sola pasada por la red, lo que los hace más rápidos. YOLO (You Only Look Once) es un ejemplo popular de este tipo.

# Conceptos Previos - Hardware NVIDIA Jetson

NVIDIA Jetson es una familia de System-on-Chip (SoC) diseñados específicamente para aplicaciones de *edge computing* e IA. Estos dispositivos integran una CPU y una GPU NVIDIA, ofreciendo un equilibrio entre rendimiento y eficiencia energética. Esto los hace ideales para aplicaciones donde el consumo de energía es un factor crítico, como en dispositivos móviles, robótica y sistemas de visión artificial.

El Jetpack SDK de NVIDIA proporciona un conjunto completo de herramientas para desarrollar aplicaciones de IA en los dispositivos Jetson. Incluye bibliotecas, APIs y herramientas de desarrollo que facilitan la implementación y optimización de modelos de *deep learning*. Una de las herramientas más importantes es TensorRT, un optimizador y *runtime* de alto rendimiento para inferencia de *deep learning*. TensorRT permite optimizar los modelos entrenados para que se ejecuten de manera eficiente en la GPU NVIDIA, reduciendo la latencia y aumentando el *throughput*.


# Conceptos Previos - Seguimiento de objetos (MOT)

El seguimiento de objetos es una técnica que permite identificar y seguir la trayectoria de múltiples objetos a lo largo del tiempo en una secuencia de imágenes o un vídeo. A diferencia de la detección de objetos, que solo identifica los objetos en un fotograma individual, el seguimiento de objetos mantiene la identidad de cada objeto a lo largo del tiempo.

El seguimiento de objetos generalmente combina la detección de objetos con el análisis de movimiento. Primero, se detectan los objetos en cada fotograma utilizando un detector de objetos. Luego, se utilizan algoritmos de seguimiento para asociar las detecciones de un fotograma con las detecciones del fotograma anterior, creando así una trayectoria para cada objeto.

BYTETrack es un algoritmo de seguimiento popular que realiza el seguimiento de objetos a partir de las detecciones de un detector de objetos. BYTETrack se destaca por su capacidad para manejar oclusiones y cambios en la apariencia de los objetos, lo que lo hace adecuado para aplicaciones en entornos complejos.

# Propuesta de solución

La propuesta de solución consiste en un sistema con la siguiente arquitectura:

1. **Entrada de vídeo:** Un sensor, en este caso una cámara, captura un vídeo en tiempo real.

2. **Detección de objetos:** Se utiliza un modelo de detección de objetos para realizar la inferencia del frame actual del vídeo.

3. **Seguimiento de objetos:** Se aplica un algoritmo de seguimiento de objetos para mantener la identidad de los objetos detectados a lo largo del tiempo.

4. **Escritura de resultados:** Se realizan anotaciones en el frame actual del vídeo, mostrando los objetos detectados y sus trayectorias. También se pueden realizar acciones basadas en los resultados de la detección, como alertas, análisis de datos o activación de actuadores.

Todo el proceso se ejecuta en tiempo real en el edge, sobre un dispositivo NVIDIA Jetson, en este caso, la Jetson AGX Xavier, Jetson AGX Orin o Jetson Orin Nano.

# Desarrollo de la solución - Entrenamiento y validación de modelos

Se ha llevado a cabo la creación de un conjunto de datos compuesto por imágenes de canicas de distintos colores, incluyendo tanto ejemplares sin defectos como con diversas anomalías visibles. A partir de este conjunto, se entrenaron modelos de redes neuronales convolucionales (CNN) con el objetivo de detectar automáticamente dichos defectos. Posteriormente, se procedió a la validación y ajuste de hiperparámetros clave, con el fin de mejorar la precisión de los modelos. Finalmente, los modelos entrenados fueron exportados al formato TensorRT para optimizar su rendimiento en dispositivos con hardware NVIDIA, aprovechando su capacidad de inferencia acelerada.

# Desarrollo de la solución - Segmentación de las etapas

El sistema propuesto se divide en cuatro etapas principales: captura de vídeo, detección de objetos, seguimiento de objetos y escritura de resultados.

- El objetivo de segmentar estas etapas es permitir que cada una opere de forma independiente, mejorando así la velocidad y la eficiencia del sistema.
- Se han planteado cuatro enfoques de segmentación para optimizar el procesamiento secuencial:
  1. Segmentación por hilos
  2. Segmentación por procesos
  3. Segmentación por procesos con memoria compartida
  4. Segmentación heterogénea

## Desarrollo de la solución - Segmentación por hilos

En este enfoque:

- Cada etapa se ejecuta en un hilo separado.
- ↑ La información se comparte entre hilos mediante colas que se consumen de forma asíncrona.
- ↓ Permite un procesamiento paralelo, aunque no concurrente, debido al *Global Interpreter Lock* (GIL) de Python.

## Desarrollo de la solución - Segmentación por procesos

En esta variante:

- Cada etapa se ejecuta en un proceso independiente.
- ↑ Permite un procesamiento concurrente real, aprovechando múltiples núcleos.
- ↓ Presenta mayor latencia en la comunicación, que se realiza mediante colas implementadas sobre *pipes*.

## Desarrollo de la solución - Segmentación por procesos con memoria compartida

Este enfoque mejora el anterior al reducir la latencia:

- Cada etapa se ejecuta en un proceso separado, pero todos comparten memoria.
- ↑ Permite procesamiento concurrente con menor latencia de comunicación.
- Las colas de comunicación se implementan sobre memoria compartida, evitando la sobrecarga de los *pipes*.


## Desarrollo de la solución - Segmentación heterogénea

En esta segmentación avanzada:

- Se pueden utilizar hilos o procesos, con o sin memoria compartida.
- La etapa de detección se ejecuta en la GPU o las DLA (Deep Learning Accelerators) del dispositivo NVIDIA Jetson.
- ↑ Permite, en teoría, aprovechar todos los recursos del sistema para maximizar el rendimiento.
- ↓ Los modelos no siempre se ejecutan completamente en la DLA, lo que puede limitar el rendimiento óptimo esperado.


# Desarrollo de la solución - Prueba de concepto

Como prueba simple de concepto, se ha construido una pequeña cinta transportadora accionada mediante un motor. El sistema desarrollado captura vídeo de la cinta en funcionamiento y es capaz de detectar canicas de distintos colores, identificando además posibles defectos en ellas.

El objetivo de esta prueba es demostrar que el sistema es capaz de detectar objetos en movimiento y clasificarlos en tiempo real, validando así la viabilidad del enfoque propuesto en un entorno controlado.

# Resultados

## Evaluación del rendimiento del sistema

Para evaluar el rendimiento del sistema, existen diferentes variables de configuración a considerar:

- **Cantidad de objetos:** Número de objetos a detectar en el vídeo.
- **Tipo de segmentación:** Método utilizado para dividir las etapas del sistema.
- **Tipo y tamaño del modelo:** Arquitectura utilizada para la detección de objetos, junto con su complejidad.
- **Precisión del modelo:** Nivel de precisión numérica empleado (FP32, FP16, INT8).
- **Modo de energía del dispositivo:** Configuración energética del hardware Jetson.
- **Modelo de dispositivo Jetson:** Versión específica del hardware utilizado.

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
  Se realizó un análisis exhaustivo de redes neuronales convolucionales (CNNs), aceleradores hardware y plataformas NVIDIA Jetson. Se abordaron con éxito los desafíos de compatibilidad y configuración asociados a arquitecturas ARM64.

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


