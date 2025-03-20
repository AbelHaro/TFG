"""
Módulo de excepciones personalizadas para el pipeline de detección y tracking.

Define excepciones específicas para diferentes tipos de errores que pueden ocurrir
durante la ejecución del pipeline, permitiendo un manejo más preciso de los errores.
"""

class PipelineError(Exception):
    """Clase base para excepciones del pipeline."""
    pass

class VideoError(PipelineError):
    """Excepciones relacionadas con la manipulación de video."""
    pass

class VideoOpenError(VideoError):
    """Error al abrir un archivo de video."""
    def __init__(self, video_path: str, message: str = None):
        self.video_path = video_path
        self.message = message or f"No se pudo abrir el archivo de video: {video_path}"
        super().__init__(self.message)

class VideoWriteError(VideoError):
    """Error al escribir un archivo de video."""
    def __init__(self, output_path: str, message: str = None):
        self.output_path = output_path
        self.message = message or f"Error al escribir el archivo de video: {output_path}"
        super().__init__(self.message)

class ModelError(PipelineError):
    """Excepciones relacionadas con el modelo de detección."""
    pass

class ModelLoadError(ModelError):
    """Error al cargar el modelo."""
    def __init__(self, model_path: str, message: str = None):
        self.model_path = model_path
        self.message = message or f"No se pudo cargar el modelo desde: {model_path}"
        super().__init__(self.message)

class InferenceError(ModelError):
    """Error durante la inferencia del modelo."""
    def __init__(self, message: str = None):
        self.message = message or "Error durante la inferencia del modelo"
        super().__init__(self.message)

class TrackerError(PipelineError):
    """Excepciones relacionadas con el tracker."""
    pass

class TrackerInitError(TrackerError):
    """Error al inicializar el tracker."""
    def __init__(self, message: str = None):
        self.message = message or "Error al inicializar el tracker"
        super().__init__(self.message)

class QueueError(PipelineError):
    """Excepciones relacionadas con las colas de procesamiento."""
    pass

class QueueTimeoutError(QueueError):
    """Timeout al esperar datos en una cola."""
    def __init__(self, queue_name: str, timeout: float):
        self.queue_name = queue_name
        self.timeout = timeout
        self.message = f"Timeout después de {timeout}s esperando datos en la cola {queue_name}"
        super().__init__(self.message)

class QueueFullError(QueueError):
    """Cola llena al intentar insertar datos."""
    def __init__(self, queue_name: str):
        self.queue_name = queue_name
        self.message = f"Cola {queue_name} llena al intentar insertar datos"
        super().__init__(self.message)

class HardwareError(PipelineError):
    """Excepciones relacionadas con el hardware."""
    pass

class GPUError(HardwareError):
    """Error relacionado con la GPU."""
    def __init__(self, message: str = None):
        self.message = message or "Error al acceder o utilizar la GPU"
        super().__init__(self.message)

class DLAError(HardwareError):
    """Error relacionado con el DLA."""
    def __init__(self, dla_id: int, message: str = None):
        self.dla_id = dla_id
        self.message = message or f"Error al acceder o utilizar DLA{dla_id}"
        super().__init__(self.message)

class ConfigError(PipelineError):
    """Excepciones relacionadas con la configuración."""
    pass

class InvalidConfigError(ConfigError):
    """Configuración inválida."""
    def __init__(self, param: str, value: str, message: str = None):
        self.param = param
        self.value = value
        self.message = message or f"Valor inválido para el parámetro {param}: {value}"
        super().__init__(self.message)

class TCPError(PipelineError):
    """Excepciones relacionadas con la comunicación TCP."""
    pass

class TCPConnectionError(TCPError):
    """Error en la conexión TCP."""
    def __init__(self, host: str, port: int, message: str = None):
        self.host = host
        self.port = port
        self.message = message or f"Error en la conexión TCP a {host}:{port}"
        super().__init__(self.message)

class TCPSendError(TCPError):
    """Error al enviar datos por TCP."""
    def __init__(self, message: str = None):
        self.message = message or "Error al enviar datos por TCP"
        super().__init__(self.message)

class FileError(PipelineError):
    """Excepciones relacionadas con operaciones de archivos."""
    pass

class FileWriteError(FileError):
    """Error al escribir un archivo."""
    def __init__(self, file_path: str, message: str = None):
        self.file_path = file_path
        self.message = message or f"Error al escribir el archivo: {file_path}"
        super().__init__(self.message)

class FileReadError(FileError):
    """Error al leer un archivo."""
    def __init__(self, file_path: str, message: str = None):
        self.file_path = file_path
        self.message = message or f"Error al leer el archivo: {file_path}"
        super().__init__(self.message)