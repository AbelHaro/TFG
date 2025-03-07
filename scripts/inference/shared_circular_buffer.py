import pickle
import numpy as np
from multiprocessing import shared_memory, Value, Lock, Condition


class SharedCircularBuffer:
    def __init__(self, queue_size=10, max_item_size=1, name=None):
        """
        Inicializa un buffer circular en memoria compartida.

        :param queue_size: Número máximo de elementos en la cola.
        :param max_item_size: Tamaño máximo en MB de cada elemento.
        :param name: Nombre de la memoria compartida (None para crear una nueva).
        """
        self.queue_size = queue_size

        if max_item_size not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            raise ValueError(
                "El tamaño máximo del item debe ser 1, 2, 4, 8, 16, 32, 64, 128, 256 o 512 MB."
            )

        self.max_item_size = max_item_size * 1024 * 1024
        self.total_size = queue_size * self.max_item_size

        if name:
            self.shm = shared_memory.SharedMemory(name=name)
        else:
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)

        self.head = Value("i", 0)  # Índice de lectura
        self.tail = Value("i", 0)  # Índice de escritura
        self.count = Value("i", 0)  # Número de elementos en la cola
        self.lock = Lock()
        self.condition = Condition(self.lock)  # Para sincronización

    def put(self, item):
        """Agrega un item a la cola en memoria compartida sin sobrescribir elementos no leídos."""
        if isinstance(item, np.ndarray):
            reshaped_item = item.reshape(-1)
            item_data = {"data": reshaped_item, "shape": item.shape}
        else:
            item_data = {"data": item, "shape": None}

        data_bytes = pickle.dumps(item_data)

        if len(data_bytes) > self.max_item_size:
            raise ValueError("El item es demasiado grande para la cola.")

        with self.condition:
            while self.count.value == self.queue_size:
                # La cola está llena, esperar hasta que haya espacio
                self.condition.wait()

            pos = (self.tail.value % self.queue_size) * self.max_item_size
            self.shm.buf[pos : pos + len(data_bytes)] = memoryview(data_bytes)

            self.tail.value = (self.tail.value + 1) % self.queue_size
            self.count.value += 1

            self.condition.notify()  # Notifica a `get` que hay un nuevo elemento disponible

    def get(self):
        """Extrae un item de la cola en memoria compartida, esperando si está vacía."""
        with self.condition:
            while self.count.value == 0:  # Esperar si la cola está vacía
                self.condition.wait()

            pos = (self.head.value % self.queue_size) * self.max_item_size
            data_bytes = bytes(self.shm.buf[pos : pos + self.max_item_size])

            self.head.value = (self.head.value + 1) % self.queue_size
            self.count.value -= 1

            self.condition.notify()  # Notifica a `put` que hay espacio disponible

        item_data = pickle.loads(data_bytes)

        if item_data["shape"] is not None:
            return np.array(item_data["data"]).reshape(item_data["shape"])

        return item_data["data"]

    def is_empty(self):
        """Retorna True si la cola está vacía."""
        with self.lock:
            return self.count.value == 0

    def is_full(self):
        """Retorna True si la cola está llena."""
        with self.lock:
            return self.count.value == self.queue_size

    def close(self):
        """Cierra la memoria compartida."""
        self.shm.close()

    def unlink(self):
        """Libera la memoria compartida (solo debe llamarse una vez)."""
        self.shm.unlink()
