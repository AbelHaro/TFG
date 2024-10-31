#!/bin/bash

# Definir la ruta del volumen en el host y en el contenedor
HOST_VOLUME=/home/abelharo/TFG
CONTAINER_VOLUME=/TFG

# Configurar el acceso a la pantalla de la computadora host si vas a usar GUI
xhost +local:root

# Construir la imagen Docker desde el Dockerfile en el directorio actual
sudo docker build -t custom-ultralytics .

# Remover contenedor existente si lo hubiera
sudo docker rm --force use-gpu

# Comando Docker base para correr la imagen de Ultralytics con Jetson y montar el volumen
DOCKER_CMD="sudo docker run -dit \
    --name use-gpu \
    --ipc=host \
    --gpus all \
    --memory=32g \
    --runtime=nvidia \
    -v \"$HOST_VOLUME:$CONTAINER_VOLUME\" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix"

# Revisar si el flag -cam está presente
if [ "$1" == "-cam" ]; then
    DOCKER_CMD="$DOCKER_CMD --device=/dev/video0:/dev/video0"
fi

# Agregar el nombre de la imagen
DOCKER_CMD="$DOCKER_CMD custom-ultralytics"

# Ejecutar el comando Docker
eval $DOCKER_CMD

