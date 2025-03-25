
# TFG

## Título

(1) Desarrollo de una aplicación para la detección de defectos en objetos en movimiento mediante Redes Neuronales Profundas con optimizaciones específicas para hardware NVIDIA.

(2) Detección de defectos en objetos en movimiento mediante Redes Neuronales Profundas con optimizaciones específicas para hardware NVIDIA.

# Desarrollo de aplicación para detección de defectos en objetos en movimiento mediante Deep Learning

Este trabajo de fin de grado abarca el desarrollo de un sistema de visión artificial para la detección de defectos en objetos en movimiento, utilizando Redes Neuronales Profundas (Deep Neural Networks) con optimizaciones específicas para hardware NVIDIA.

El proyecto implementa un sistema completo de detección y seguimiento de objetos en tiempo real, utilizando YOLOv11 como base y aplicando diversas técnicas de optimización. Se ha desarrollado un pipeline modular que incluye procesamiento paralelo, uso de memoria compartida y optimizaciones específicas para hardware NVIDIA como TensorRT y DLA.

El sistema incorpora técnicas avanzadas como SAHI (Slicing Aided Hyper Inference) para mejorar la detección en objetos pequeños. Todo el sistema está diseñado para funcionar en tiempo real, con especial énfasis en la optimización del rendimiento manteniendo la precisión en la detección.

Se ha creado además un conjunto de datos propio para entrenamiento y evaluación, junto con herramientas específicas para la validación y medición del rendimiento del sistema en diferentes configuraciones de hardware.

## Características principales

- Implementación de modelos de Deep Learning para detección de defectos
- Optimizaciones a nivel de modelo y ejecución para mejorar el rendimiento
- Dataset propio de imágenes y videos para evaluación del sistema
- Optimizaciones específicas para hardware NVIDIA
- Análisis de objetos en movimiento en tiempo real

## Objetivos

El proyecto se centra en maximizar la eficiencia y velocidad de detección manteniendo la precisión, mediante:

1. Optimización de arquitectura del modelo
2. Mejoras en el rendimiento de ejecución
3. Aprovechamiento del hardware NVIDIA
4. Validación con datos reales

Este sistema está diseñado para aplicaciones industriales que requieren inspección visual automatizada de alta velocidad

## Resumen largo

Este trabajo de fin de grado abarca el desarrollo de un sistema de visión artificial para la detección de defectos en objetos en movimiento, utilizando Redes Neuronales Profundas (Deep Neural Networks) con optimizaciones específicas para hardware NVIDIA. Aprovechar al máximo las capacidades de estos dispositivos no es una tarea sencilla, ya que depende de múltiples factores, como la gestión eficiente de la memoria, la complejidad del modelo neuronal y las limitaciones computacionales del hardware específico. Por ello, este trabajo se centra en identificar los cuellos de botella en el diseño y optimizar el flujo de procesamiento para maximizar el rendimiento del sistema en tiempo real.

Para ello, se utilizan dispositivos de la serie Nvidia Jetson, diseñados para tareas de inteligencia artificial en el borde (edge computing). Existen principalmente dos formas de optimizar la ejecución de redes neuronales en este hardware: mediante TensorRT, que permite la aceleración del modelo utilizando técnicas como la cuantización y la fusión de capas, y/o utilizando el Deep Learning Accelerator (DLA), un motor de hardware especializado para la inferencia eficiente.

El sistema implementado permite la detección y seguimiento de objetos en tiempo real mediante el uso de YOLOv11, aplicando diversas técnicas de optimización. Para mejorar la eficiencia, se ha desarrollado un pipeline modular que incorpora procesamiento paralelo, uso de memoria compartida y optimizaciones específicas para hardware NVIDIA, como TensorRT y DLA. La detección de objetos pequeños se ha mejorado mediante la técnica SAHI (Slicing Aided Hyper Inference). De esta forma, el sistema mantiene un equilibrio entre precisión y rendimiento, asegurando una ejecución en tiempo real.

Para comparar las técnicas utilizadas, se ha implementado una aplicación que evalúa el rendimiento obtenido, la utilización de recursos y el consumo de energía.
