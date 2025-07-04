# Usa la imagen base de Ultralytics para Jetson con JetPack 5
#FROM ultralytics/ultralytics:latest-jetson-jetpack5
FROM ultralytics/ultralytics:8.3.38-jetson-jetpack5

# Actualiza los repositorios e instala Python 3.10 y sus dependencias
#RUN apt-get update && apt-get install -y \
#    software-properties-common && \
#   add-apt-repository ppa:deadsnakes/ppa && \
#  apt-get update && \
# apt-get install -y \
#python3.10 python3.10-venv python3.10-dev

# Actualiza e instala PyTorch y torchvision
#RUN pip install --upgrade pip && \
#  pip install -U torch torchvision

RUN sudo apt-get update
RUN sudo apt-get install nano

# Instala Ultralytics
RUN pip install ultralytics

# Instala onnxruntime
#RUN pip install onnxslim==0.1.34

# Dependencia para soliconar un error de versión mayor a ultralytics/ultralytics:8.3.38-jetson-jetpack5
RUN pip install numpy==1.23.5

# Instala openpyxl
RUN pip install openpyxl

RUN pip install websockets

WORKDIR /TFG

CMD ["/bin/bash"]
