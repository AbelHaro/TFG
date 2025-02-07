#!/bin/bash

HOST_VOLUME=/home/abelharo/TFG
CONTAINER_VOLUME=/TFG

# Configurar el acceso a la pantalla de la computadora host si vas a usar GUI
xhost +local:root

# Construir la imagen
echo "Construyendo la imagen de Docker..."
sudo docker build -t custom-ultralytics . || { echo "Error al construir la imagen"; exit 1; }

# Eliminar el contenedor existente si ya está corriendo
echo "Eliminando cualquier contenedor existente con el nombre 'use-gpu'..."
sudo docker rm --force use-gpu

# Crear el comando Docker
DOCKER_CMD="sudo docker run -dit \
    --name use-gpu \
    -p 6006:6006 \
    -p 8765:8765 \
    --ipc=host \
    --privileged \
    --memory=32g \
    --gpus all \
    --runtime=nvidia \
    -v \"$HOST_VOLUME:$CONTAINER_VOLUME\" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/:/tmp"

# Revisar si el flag -cam está presente
if [ "$1" == "-cam" ]; then
    echo "Agregando soporte para cámara..."
    DOCKER_CMD="$DOCKER_CMD --device=/dev/video0:/dev/video0"
else
    echo "No se agregó soporte para cámara. Para agregarlo, reinicia el script con el flag '-cam'."
fi

# Agregar acceso a DLA (nvdla devices)
DOCKER_CMD="$DOCKER_CMD --device=/dev/nvhost-ctrl-nvdla0 --device=/dev/nvhost-ctrl-nvdla1 --device=/dev/nvhost-nvdla0 --device=/dev/nvhost-nvdla1"

# Agregar acceso para ejecutar tegrastats
DOCKER_CMD="$DOCKER_CMD \
    --device=/dev/nvhost-ctrl \
    --device=/dev/nvhost-ctrl-gpu \
    --device=/dev/nvmap \
    --device=/dev/nvhost-prof-gpu \
    -v /proc:/proc \
    -v /sys:/sys"

# Agregar el nombre de la imagen
DOCKER_CMD="$DOCKER_CMD custom-ultralytics"

# Ejecutar el comando Docker
echo "Ejecutando el contenedor..."
eval $DOCKER_CMD || { echo "Error al ejecutar el contenedor"; exit 1; }


# Verificar si tegrastats está disponible
echo "Verificando si 'tegrastats' está disponible en el contenedor..."
docker exec use-gpu which tegrastats > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "'tegrastats' no está disponible en el contenedor. Copiándolo desde el host..."
        docker cp /usr/bin/tegrastats use-gpu:/usr/bin/
        docker exec use-gpu chmod +x /usr/bin/tegrastats

    # Ejecutar tegrastats dentro del contenedor
    #echo "Ejecutando 'tegrastats' dentro del contenedor..."
    #docker exec -it use-gpu tegrastats || { echo "Error al ejecutar 'tegrastats' dentro del contenedor"; exit 1; }
fi
