# Trabajo hecho hasta el momento

Aqui se detallan los avances realizados hasta el momento en el proyecto de fin de grado

## SOBRE EL DATASET

- [x] Generación de un dataset de 400 imágenes de canicas con 8 clases diferentes {negra, blanca, azul, verde, negra defectuosa, blanca defectuosa, azul defectuosa, verde defectuosa}

## SOBRE REDES NEURONALES

- [x] Entrenamiento de YOLO11 en sus diferentes talla {n, s, m, l, x} con el dataset de canicas
- [x] Programa para hacer prunning de los modelos entrenados
- [x] Exportación de Pytorch a ONNX y TensorRT en {FP32, FP16, INT8} y para {GPU, DLA0 y DLA1}
- [x] En TensorRT, diferentes argumentos de exportación como el batch

## SOBRE HARDWARE

- [x] Exploración de diferentes aproximaciones para el pipeline de inferencia {secuencial, segmentación por hilos, segmentación por procesos (memoria distribuida) y segmentación por procesos (memoria compartida), multihardware (GPU, DLA0 y DLA1)} para el aprovechamiento de los recursos
- [x] Programa para hacer pruebas de inferencia en diferentes aproximaciones
- [x] Análisis de consumo energético en diferentes aproximaciones
- [x] Programa para el análisis de consumo sobre un modelo sencillo completamente en DLA frente a un modelo completamente en GPU para el dataset CIFAR10

## SOBRE EL PIPELINE DE INFERENCIA

- [x] Uso de BYTETrack para el tracking de los de los objetos mediante un ID
- [x] Algoritmo de memoria para asignar la clase mediante el ID
- [x] Uso de SAHI para la mejora de la precisión en la inferencia
- [x] Comunicación mediante TCP entre la máquina de inferencia y el microcontrolador de la cinta transportadora
- [x] Uso de ArUco para la estimación de la velocidad de los objetos sobre la cinta transportadora

## SOBRE LA CINTA TRANSPORTADORA

- [x] Comunicación mediante TCP con la máquina de inferencia
- [x] Programación del servo para el control de la cinta transportadora
- [ ] Construcción de la cinta transportadora

## VARIABLES A MODIFICAR PARA REALIZAR PRUEBAS

- Cantidad de objetos en la imagen [0..desconocido] (idea, tener 2 vídeos, uno en el que la cantidad de objetos sea fija y otro en el que la cantidad de objetos sea variable) casos 2
- Tamaño del video de entrada [(640,640), (1080,1080)(SAHI batch 4), (1920,1080)(SAHI batch 8)] casos 3
- Talla del modelo de YOLO11 [n, s, m, l] casos 4
- Modo de energía de la Jetson [10W..50W] dependiendo de la Jetson casos 3
- Número de procesadores a usar [1..12] dependiendo de la Jetson casos 2
- Jetson a usar [Nano, Xavier, Orin] casos 3
- Modo de segmentación [secuencial, hilos, procesos (memoria distribuida), procesos (memoria compartida), procesos multihardware (memoria compartida)] casos 5
- Modo de precisión [FP32, FP16, INT8] casos 1, descartar FP32 e INT8
- Modo de hardware [GPU, DLA] casos 2

Total de combinaciones = (2 x 3 x 4 x 3 x 2 x 3 x 5 x 1 x 2) = 4320 casos, demasiados casos para realizar pruebas, demasiadas combinaciones

### 3 opciones para poder realizar pruebas de rendimiento

(1) Usar un script que contemple todos los casos y haga las ejecuciones sobre un vídeo pregrabado
    -Problema: Cada caso ejecutará el programa en un tiempo diferente, por lo tanto no se adaptará a las necesidades de la cinta transportadora, donde el programa debe ejecutarse con un framerate determinado por la cámara.

(2) Realizar las pruebas sobre la cinta transportadora
    -Problema: Demasiados casos a contemplar, por lo que se necesitaría un tiempo excesivo para realizar todas las pruebas
    -Problema: Los vídeos que se procesen no serán los mismos, por lo que no se podrán comparar los resultados
    -Problema: Aún no se ha construido la cinta transportadora

(3) Usar un script que contemple todos los casos y haga las ejecuciones sobre un vídeo pregrabado simulando que la cámara emite los frames a 30fps
