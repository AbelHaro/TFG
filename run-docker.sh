#!/bin/bash

HOST_VOLUME=/home/abelharo/TFG
CONTAINER_VOLUME=/TFG

# Configurar el acceso a la pantalla de la computadora host si vas a usar GUI
xhost +local:root

sudo docker build -t custom-ultralytics .

sudo docker rm --force use-gpu

DOCKER_CMD="sudo docker run -dit \
    --name use-gpu \
    -p 6006:6006 \
    --ipc=host \
    --memory=32g \
    --gpus all \
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

