import PyNvVideoCodec as nvc
import os

# Ruta del archivo de video
enc_file_path = "../datasets_labeled/videos/assert.mp4"

# Comprobamos si el archivo existe
if not os.path.isfile(enc_file_path):
    raise FileNotFoundError("El archivo de video no existe")

# Crear el demuxer (desmultiplexor) para extraer los paquetes de video
demuxer = nvc.CreateDemuxer(filename=enc_file_path)

# Crear el decodificador (usando H.264 como códec)
try:
    decoder = nvc.CreateDecoder(
        gpuid=0,
        codec=nvc.cudaVideoCodec.H264,  # Especificar el códec H.264
        cudacontext=0,
        cudastream=0,
        enableasyncallocations=True,  # Usar este parámetro en lugar de 'usedevicememory'
    )
except Exception as e:
    print(f"Error al crear el decodificador: {e}")
    exit(1)

# Proceso de decodificación: iterar sobre los paquetes del demuxer
frame_counter = 0
for packet in demuxer:
    # Decodificar el paquete
    for decoded_frame in decoder.Decode(packet):
        frame_counter += 1
        print(f"Frame {frame_counter} decodificado")

        # Acceder a los datos del frame decodificado
        luma_base_addr = decoded_frame.lumaBaseAddress()  # Dirección base de la luma
        chroma_base_addr = (
            decoded_frame.chromaBaseAddress()
        )  # Dirección base del croma (si es aplicable)

        # Obtener el tamaño del frame
        frame_size = decoded_frame.framesize()

        # Crear un arreglo de bytes a partir de la memoria del frame (como ejemplo)
        new_array = bytearray(luma_base_addr[:frame_size])

        # Aquí puedes procesar el frame según sea necesario
        # Ejemplo: Mostrar el tamaño del arreglo de bytes (que representa el frame)
        print(f"Tamaño del frame: {len(new_array)} bytes")

        # También podrías hacer algo con el frame, como almacenarlo o procesarlo
        # Por ejemplo, podrías convertirlo a una imagen con OpenCV o guardarlo
        # (Este paso depende de tu necesidad específica)

        # Para finalizar el proceso si no hay más frames:
        if not decoded_frame:
            print("No more frames to decode.")
            break
