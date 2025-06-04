import pickle
import numpy as np
from multiprocessing import shared_memory, Value, Lock, Condition


class SharedCircularBuffer:
    def __init__(self, queue_size=10, max_item_size=1, name=None):
        """
        Initializes a circular buffer in shared memory.

        :param queue_size: Maximum number of elements in the queue.
        :param max_item_size: Maximum size in MB of each element.
        :param name: Name of the shared memory (None to create a new one).
        """
        self.queue_size = queue_size

        if max_item_size not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            raise ValueError(
                "The maximum item size must be 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512 MB."
            )

        self.max_item_size = max_item_size * 1024 * 1024
        self.total_size = queue_size * self.max_item_size

        if name:
            self.shm = shared_memory.SharedMemory(name=name)
        else:
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)

        self.head = Value("i", 0)  # Read index
        self.tail = Value("i", 0)  # Write index
        self.count = Value("i", 0)  # Number of elements in the queue
        self.lock = Lock()
        self.condition = Condition(self.lock)  # For synchronization

    def put(self, item):
        """Adds an item to the queue in shared memory without overwriting unread elements."""
        if isinstance(item, np.ndarray):
            reshaped_item = item.reshape(-1)
            item_data = {"data": reshaped_item, "shape": item.shape}
        else:
            item_data = {"data": item, "shape": None}

        data_bytes = pickle.dumps(item_data)

        if len(data_bytes) > self.max_item_size:
            raise ValueError("The item is too large for the queue.")

        with self.condition:
            while self.count.value == self.queue_size:
                # The queue is full, wait until there is space
                self.condition.wait()

            pos = (self.tail.value % self.queue_size) * self.max_item_size
            self.shm.buf[pos : pos + len(data_bytes)] = memoryview(data_bytes)

            self.tail.value = (self.tail.value + 1) % self.queue_size
            self.count.value += 1

            self.condition.notify()  # Notify `get` that a new element is available
            
    def put_nowait(self, item):
        """Adds an item to the queue in shared memory. If the queue is full, returns False without adding the item."""
        if isinstance(item, np.ndarray):
            reshaped_item = item.reshape(-1)
            item_data = {"data": reshaped_item, "shape": item.shape}
        else:
            item_data = {"data": item, "shape": None}

        data_bytes = pickle.dumps(item_data)

        if len(data_bytes) > self.max_item_size:
            raise ValueError("The item is too large for the queue.")

        with self.condition:
            if self.count.value >= self.queue_size:
                return False  # Queue is full, cannot add the item

            pos = (self.tail.value % self.queue_size) * self.max_item_size
            self.shm.buf[pos : pos + len(data_bytes)] = memoryview(data_bytes)
            
            self.tail.value = (self.tail.value + 1) % self.queue_size
            self.count.value += 1

            self.condition.notify()  # Notify `get` that a new element is available
            return True
        
        
        
        

    def get(self):
        """Extracts an item from the queue in shared memory, waiting if it is empty."""
        with self.condition:
            while self.count.value == 0:  # Wait if the queue is empty
                self.condition.wait()

            pos = (self.head.value % self.queue_size) * self.max_item_size
            data_bytes = bytes(self.shm.buf[pos : pos + self.max_item_size])

            self.head.value = (self.head.value + 1) % self.queue_size
            self.count.value -= 1

            self.condition.notify()  # Notify `put` that space is available

        item_data = pickle.loads(data_bytes)

        if item_data["shape"] is not None:
            return np.array(item_data["data"]).reshape(item_data["shape"])

        return item_data["data"]

    def is_empty(self):
        """Returns True if the queue is empty."""
        with self.lock:
            return self.count.value == 0

    def is_full(self):
        """Returns True if the queue is full."""
        with self.lock:
            return self.count.value == self.queue_size

    def close(self):
        """Closes the shared memory."""
        self.shm.close()

    def unlink(self):
        """Releases the shared memory (should only be called once)."""
        self.shm.unlink()
