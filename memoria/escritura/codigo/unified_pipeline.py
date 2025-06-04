from abc import ABC
import cv2
import torch.multiprocessing as mp
from queue import Queue
from classes.shared_circular_buffer import SharedCircularBuffer
from detection_tracking_pipeline import DetectionTrackingPipeline
import threading
import logging
from typing import Union, List, Type, Any


class UnifiedPipeline(DetectionTrackingPipeline):
    """Unified pipeline that supports different parallelization strategies.

    This class inherits from `DetectionTrackingPipeline` and provides a concrete
    implementation that allows configuring the pipeline to run in different modes:
    - 'mp_hardware': Uses multiple processes and shared memory queues,
                     optimized for scenarios with multiple hardware accelerators
                     (e.g. GPU, DLA0, DLA1).
    - 'threads': Uses threads for concurrency within the same process.
    - 'mp_shared_memory': Uses multiple processes with shared memory queues
                          (SharedCircularBuffer).
    - Default (any other string): Uses multiple processes with standard
                                  `torch.multiprocessing` queues.
    """

    def __init__(
        self,
        video_path: str,
        model_path: str,
        output_video_path: str,
        output_times: str,
        parallel_mode: str,
        is_tcp: bool = False,
        sahi: bool = False,
        max_fps: int = None,
        dla0_model: str = None,
        dla1_model: str = None,
    ):
        """Initializes the unified pipeline.

        Args:
            video_path: Path to the input video file.
            model_path: Path to the main detection model (e.g. for GPU).
            output_video_path: Path to save the processed video with detections.
            output_times: Path to the CSV file to save processing times.
            parallel_mode: Parallelization strategy to use.
            is_tcp: Boolean to enable TCP communication (e.g. to notify defects).
            sahi: Boolean to enable SAHI (Slice-Aided Hyper Inference).
            max_fps: Optional. Limits the frames per second of processing.
            dla0_model: Optional. Path to the model for DLA0 accelerator (if `parallel_mode` is 'mp_hardware').
            dla1_model: Optional. Path to the model for DLA1 accelerator (if `parallel_mode` is 'mp_hardware').
        """
        self.parallel_mode = parallel_mode
        # Determines if workers will be processes or threads based on the parallelization mode.
        self.is_process = parallel_mode != "threads"
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.is_tcp = is_tcp
        self.sahi = sahi
        self.max_fps = max_fps
        self.dla0_model = dla0_model
        self.dla1_model = dla1_model
        # mh_num is used to indicate to capture_frames how many 'None' signals to send
        # when finishing, so that all frame_queue consumers terminate.
        self.mh_num = 1 # Default: one frame consumer (process_frames), if using mp_hardware, it would be 3 (GPU + DLA0 + DLA1)
        
        self._initialize_queues()
        self._initialize_events()
        # Initializes shared memory for object tracking.
        # This memory is used by `draw_and_write_frames` and updated by `update_memory`.
        self.memory = {}

    def _initialize_queues(self):
        """Initializes communication queues between pipeline stages.

        The choice of queue type (standard Queue, mp.Queue, SharedCircularBuffer)
        and its size is based on `parallel_mode` and whether `max_fps` is defined.
        `SharedCircularBuffer` is used for modes with explicit shared memory.
        """
        # Queue size: 1 if max_fps is limited (to avoid accumulation), otherwise 10.
        queue_size = 1 if self.max_fps else 10

        if self.parallel_mode == "mp_hardware":
            # Multiple detection queues for different hardware (GPU, DLA0, DLA1)
            # and a common frame queue. All use SharedCircularBuffer.
            logging.info("mp_hardware mode: Using SharedCircularBuffer for all queues.")
            self.frame_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16) # Arbitrary item size
            self.detection_queue_GPU = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.detection_queue_DLA0 = SharedCircularBuffer(
                queue_size=queue_size, max_item_size=16
            )
            self.detection_queue_DLA1 = SharedCircularBuffer(
                queue_size=queue_size, max_item_size=16
            )
            self.tracking_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.times_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.mh_num = 3  # One frame capturer feeds 3 frame processors (GPU, DLA0, DLA1)
        
        elif self.parallel_mode == "threads":
            # Standard `queue.Queue` queues for communication between threads.
            logging.info("threads mode: Using queue.Queue.")
            self.frame_queue = Queue(maxsize=queue_size)
            self.detection_queue = Queue(maxsize=queue_size)
            self.tracking_queue = Queue(maxsize=queue_size)
            self.times_queue = Queue(maxsize=queue_size)
        
        elif self.parallel_mode == "mp_shared_memory":
            # `SharedCircularBuffer` queues for multiprocessing with shared memory.
            logging.info("mp_shared_memory mode: Using SharedCircularBuffer.")
            self.frame_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.detection_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.tracking_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
            self.times_queue = SharedCircularBuffer(queue_size=queue_size, max_item_size=16)
        
        else: # Default mode: standard multiprocessing
            # `mp.Queue` queues from `torch.multiprocessing`.
            # If max_fps is defined, the frame queue has size 1 to process frame by frame.
            logging.info("Standard multiprocessing mode: Using mp.Queue.")
            self.frame_queue = (
                mp.Queue(maxsize=1) if self.max_fps else mp.Queue(maxsize=queue_size)
            )
            self.detection_queue = mp.Queue(maxsize=queue_size)
            self.tracking_queue = mp.Queue(maxsize=queue_size)
            self.times_queue = mp.Queue(maxsize=queue_size)

    def _initialize_events(self):
        """Initializes synchronization events.

        All events are from `torch.multiprocessing.Event` regardless of mode,
        since they can be shared between processes if necessary,
        and also work correctly with threads.
        - `stop_event`: Signal to stop all pipeline workers.
        - `t1_start`: Signal to synchronize the start of timing and
                      certain operations after model initialization.
        - `tcp_event`: Signal to synchronize TCP operations.
        - `mp_stop_event`: Signal for workers to wait before exiting,
                           allowing an orderly shutdown.
        """
        # Multiprocessing events are used for all modes for consistency
        # and because they work for both processes and threads.
        self.stop_event = mp.Event()
        self.t1_start = mp.Event()
        self.tcp_event = mp.Event()
        self.mp_stop_event = mp.Event() # Used for workers to wait before os._exit if they are processes

    def _get_worker_class(self) -> Type[Union[mp.Process, threading.Thread]]:
        """Determines the base class for workers (process or thread).

        Returns:
            The `threading.Thread` class if `parallel_mode` is 'threads',
            otherwise, `torch.multiprocessing.Process`.
        """
        if self.parallel_mode == "threads":
            return threading.Thread
        return mp.Process

    def _create_workers(self) -> List[Union[mp.Process, threading.Thread]]:
        """Creates and initializes the list of workers (processes or threads) for the pipeline.

        Each worker is an instance of the class returned by `_get_worker_class()`.
        The configuration of workers (target functions and arguments) depends
        on the `parallel_mode` and whether SAHI or multiple hardware is used.

        Returns:
            A list of Worker objects (Process or Thread) ready to be started.
        """
        Worker = self._get_worker_class()
        workers: List[Union[mp.Process, threading.Thread]] = []

        # 1. Frame Capture Worker (common to all modes)
        # This worker reads frames from the video and puts them in `frame_queue`.
        workers.append(
            Worker(
                name="CaptureWorker",
                target=self.capture_frames,
                args=(
                    self.video_path,
                    self.frame_queue,
                    self.t1_start,      # Event to start capture after others are ready
                    self.stop_event,    # Event to stop capture
                    self.tcp_event,     # Event for TCP synchronization (if is_tcp)
                    self.is_tcp,
                    self.mp_stop_event, # Event to wait before exiting (if process)
                    self.mh_num,        # Number of frame_queue consumers
                    self.is_process,    # True if the worker is a process
                    self.max_fps,       # FPS limit for capture
                ),
            )
        )

        # 2. Frame Processing Workers (Detection)
        if self.parallel_mode == "mp_hardware":
            # Multiple detection workers, one for each specified model/hardware.
            # Each consumes from `frame_queue` and produces to its `detection_queue_*`.
            hardware_setups = [
                (self.model_path, self.detection_queue_GPU, "GPU"),
                (self.dla0_model, self.detection_queue_DLA0, "DLA0"),
                (self.dla1_model, self.detection_queue_DLA1, "DLA1"),
            ]
            for model_p, detection_q, hw_name in hardware_setups:
                if model_p: # Only create the worker if a model path was provided
                    workers.append(
                        Worker(
                            name=f"ProcessWorker_{hw_name}",
                            target=self.process_frames_sahi if self.sahi else self.process_frames,
                            args=(
                                self.frame_queue,
                                detection_q,
                                model_p,
                                self.t1_start,      # Signals when the model is ready
                                self.mp_stop_event,
                                self.is_process,
                            ),
                        )
                    )
            
            # Multi-Hardware specific Tracking Worker
            # Consumes from all `detection_queue_*` and produces to `tracking_queue`.
            workers.append(
                Worker(
                    name="TrackingWorker_MH",
                    target=self.tracking_frames_multihardware,
                    args=(
                        self.detection_queue_GPU,
                        self.detection_queue_DLA0,
                        self.detection_queue_DLA1,
                        self.tracking_queue,
                        self.mp_stop_event,
                        self.is_process,
                    ),
                )
            )
        else:
            # Single Frame Processing Worker (Detection)
            # Consumes from `frame_queue` and produces to `detection_queue`.
            workers.append(
                Worker(
                    name="ProcessWorker",
                    target=self.process_frames_sahi if self.sahi else self.process_frames,
                    args=(
                        self.frame_queue,
                        self.detection_queue,
                        self.model_path,
                        self.t1_start,
                        self.mp_stop_event,
                        self.is_process,
                    ),
                )
            )

            # Standard Tracking Worker
            # Consumes from `detection_queue` and produces to `tracking_queue`.
            workers.append(
                Worker(
                    name="TrackingWorker",
                    target=self.tracking_frames,
                    args=(
                        self.detection_queue,
                        self.tracking_queue,
                        self.mp_stop_event,
                        self.is_process,
                    ),
                )
            )

        # 3. Workers common to all modes (Drawing, CSV Writing, Hardware Usage)
        
        # Worker to draw detections/tracking on frames and write the output video.
        # Consumes from `tracking_queue` and can interact with TCP.
        workers.append(
            Worker(
                name="DrawWriteWorker",
                target=self.draw_and_write_frames,
                args=(
                    self.tracking_queue,
                    self.times_queue,       # To send end signal to write_to_csv
                    self.output_video_path,
                    self.CLASSES,           # Class definitions for drawing
                    self.memory,            # Tracking memory for consistency
                    self.COLORS,            # Colors for classes
                    self.stop_event,        # To stop if other parts fail
                    self.tcp_event,         # For TCP synchronization
                    self.is_tcp,
                    self.mp_stop_event,
                    self.is_process,
                ),
            )
        )
        return workers

    def _cleanup(self):
        """Cleans up resources used by the pipeline, specifically the queues.

        For modes that use `SharedCircularBuffer` ('mp_shared_memory', 'mp_hardware'),
        it is necessary to explicitly close (`close()`) and unlink (`unlink()`)
        the shared memory buffers to free system resources.
        Standard queues (Queue, mp.Queue) are managed by the garbage collector
        or when processes/threads terminate and do not require this manual cleanup.
        """
        logging.info("Starting resource cleanup...")
        if self.parallel_mode in ["mp_shared_memory", "mp_hardware"]:
            logging.info(f"{self.parallel_mode} mode: Cleaning SharedCircularBuffer queues.")
            # List of all queues that are SharedCircularBuffer
            queues_to_clean = [self.frame_queue, self.tracking_queue, self.times_queue]

            if self.parallel_mode == "mp_hardware":
                queues_to_clean.extend(
                    [
                        self.detection_queue_GPU,
                        self.detection_queue_DLA0,
                        self.detection_queue_DLA1,
                    ]
                )
            else: # mp_shared_memory
                queues_to_clean.append(self.detection_queue)

            for i, queue_buffer in enumerate(queues_to_clean):
                try:
                    if isinstance(queue_buffer, SharedCircularBuffer): # Double check just in case
                        logging.debug(f"Closing and unlinking buffer {i}...")
                        queue_buffer.close()
                        queue_buffer.unlink()
                        logging.debug(f"Buffer {i} cleaned.")
                    else:
                        logging.warning(f"Expected SharedCircularBuffer in cleanup, got {type(queue_buffer)}")
                except Exception as e:
                    logging.error(f"Error cleaning buffer {i}: {e}")
        else:
            logging.info(f"{self.parallel_mode} mode: No manual cleanup required for standard queues.")
        logging.info("Resource cleanup finished.")

    def run(self):
        """Runs the unified pipeline.

        This method orchestrates the creation, startup and termination of workers.
        Measures the total processing time from when `t1_start` is activated
        (usually after models are ready) until `stop_event`
        is activated (usually by `draw_and_write_frames` when finishing processing).
        Finally, performs resource cleanup and waits for thread termination
        if that is the parallelization mode.
        """
        logging.info(f"Starting pipeline in mode: {self.parallel_mode}")
        
        # Create and start all workers (processes or threads)
        workers = self._create_workers()
        for worker in workers:
            logging.info(f"Starting worker: {worker.name}")
            worker.start()

        # Wait for the initialization stage (e.g. model loading in process_frames)
        # to activate the t1_start event. This marks the actual start of measurable processing.
        logging.info("Waiting for t1_start signal (models ready/measurement start)...")
        self.t1_start.wait()
        logging.info("t1_start signal received.")

        t_start_processing = cv2.getTickCount()

        # Wait for the pipeline to complete its main task.
        # `stop_event` is usually activated by the last worker in the chain
        # (draw_and_write_frames) when there are no more frames to process.
        logging.info("Pipeline running. Waiting for stop_event signal (end of processing)...")
        self.stop_event.wait()
        logging.info("stop_event signal received.")

        t_end_processing = cv2.getTickCount()

        # Clean up resources (e.g. shared memory queues)
        # It's important to do this before processes terminate completely.
        self._cleanup()

        # Calculate and display performance statistics
        tiempo_total_segundos = (t_end_processing - t_start_processing) / cv2.getTickFrequency()

        # Get the total number of frames from the video to calculate average FPS.
        # I assume that get_total_frames is a method (possibly static or instance)
        # that reads video metadata.

        print(f"Total processing time: {tiempo_total_segundos:.2f} seconds")


        # Wait for all workers to finish, especially important for threads.
        # For processes, mp_stop_event and os._exit() handle their termination.
        if self.parallel_mode == "threads":
            logging.info("Waiting for thread termination...")
            for worker in workers:
                if worker.is_alive(): # Only join if the thread is still alive
                    logging.info(f"Waiting for {worker.name}...")
                    worker.join(timeout=10) # Add a timeout to avoid indefinite blocking
                    if worker.is_alive():
                        logging.warning(f"Thread {worker.name} did not finish after timeout.")
                    else:
                        logging.info(f"Thread {worker.name} finished.")
        
        # Signal processes that they can terminate (if they are waiting for mp_stop_event)
        if self.is_process:
             self.mp_stop_event.set()

        print("Pipeline finished.")
