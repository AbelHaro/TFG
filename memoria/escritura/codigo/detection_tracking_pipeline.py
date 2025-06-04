from abc import ABC, abstractmethod
import cv2
import os
from argparse import Namespace
from classes.tracker_wrapper import TrackerWrapper
from lib.tcp import handle_send, tcp_server
import logging
from typing import Union, Optional
import torch.multiprocessing as mp
from classes.shared_circular_buffer import SharedCircularBuffer
from lib.constants import TIMING_FIELDS
import time



class DetectionTrackingPipeline(ABC):
    """Abstract base class for object detection and tracking pipelines.

    This class defines the structure and common functionality for various pipelines
    that integrate object detection and their subsequent tracking through video
    sequences. It allows the implementation of different parallelization strategies
    and hardware management.
    """

    CLASSES = {
        0: "negra",
        1: "blanca",
        2: "verde",
        3: "azul",
        4: "negra-d",
        5: "blanca-d",
        6: "verde-d",
        7: "azul-d",
    }

    COLORS = {
        "negra": (0, 0, 255),
        "blanca": (0, 255, 0),
        "verde": (255, 0, 0),
        "azul": (255, 255, 0),
        "negra-d": (0, 165, 255),
        "blanca-d": (255, 165, 0),
        "verde-d": (255, 105, 180),
        "azul-d": (255, 0, 255),
    }


    def update_memory(self, tracked_objects, memory, classes) -> None:
        """Updates tracking memory with detected and tracked objects.

        Maintains a record of objects, their state (defective or not),
        and their visibility throughout frames. An object is considered
        permanently defective if detected as defective during a
        consecutive number of frames defined by `PERMANENT_DEFECT_THRESHOLD`.
        Objects that have not been seen for `FRAME_AGE` frames are removed
        from memory.

        Args:
            tracked_objects: List of tracked objects in the current frame.
                             Each object is a tuple or list with information like
                             tracking ID, detected class, etc.
            memory: Dictionary that stores the state of tracked objects
                    between frames. Keys are tracking IDs.
            classes: Dictionary that maps class IDs to their names.
        """
        FRAME_AGE = 60  # Number of frames to keep an object in memory if not visible
        PERMANENT_DEFECT_THRESHOLD = (
            3  # Consecutive frames to mark as "permanent defect"
        )

        for obj in tracked_objects:
            track_id = int(obj[4])
            detected_class = classes[int(obj[6])]
            is_defective = detected_class.endswith("-d")

            if track_id in memory:
                entry = memory[track_id]

                # If already marked as permanent defect, only update its visibility
                if entry.get("permanent_defect", False):
                    entry["visible_frames"] = FRAME_AGE
                    continue

                # Updates the consecutive defect counter
                if is_defective:
                    entry["defect_counter"] = entry.get("defect_counter", 0) + 1
                else:
                    entry["defect_counter"] = 0  # Reset if not defective in this frame

                # Mark as permanent defect if it reaches the threshold
                if entry["defect_counter"] >= PERMANENT_DEFECT_THRESHOLD:
                    entry["permanent_defect"] = True
                    # The class already includes '-d', no need to reassign `detected_class` here
                    # entry["defective"] will be updated below

                # Update defective status and visibility
                entry["defective"] = entry.get("permanent_defect", False) or is_defective
                entry["visible_frames"] = FRAME_AGE
                entry["class"] = detected_class
            else:
                # New detected object
                memory[track_id] = {
                    "defective": is_defective,
                    "visible_frames": FRAME_AGE,
                    "class": detected_class,
                    "defect_counter": 1 if is_defective else 0,
                    "permanent_defect": False, # Initialize as not permanent
                }

        # Decrement visibility and remove old objects
        for track_id in list(memory):  # Iterate over a copy of the keys
            memory[track_id]["visible_frames"] -= 1
            if memory[track_id]["visible_frames"] <= 0:
                del memory[track_id]

    def capture_frames(
            self,
            video_path: str,
            frame_queue: Union[mp.Queue, SharedCircularBuffer],
            t1_start: mp.Event,
            stop_event: mp.Event,
            tcp_event: mp.Event,
            is_tcp: bool,
            mp_stop_event: Optional[mp.Event] = None,
            mh_num: int = 1,
            is_process: bool = False,
            max_frames: Optional[int] = None,
            ):
            """Captures frames from a video file and queues them for processing.

            Reads frames from a video specified by `video_path`. If `is_tcp` is True,
            waits for `tcp_event` to be set before starting capture.
            Frames are put into `frame_queue`. If `max_frames` is defined,
            attempts to maintain that FPS rate by limiting capture speed.
            When finished or if `stop_event` is set, sends `None` to the queue
            (as many times as `mh_num`) to signal the end of capture.

            Args:
                video_path: Path to the video file.
                frame_queue: Queue (multiprocessing or shared circular buffer) to send frames.
                t1_start: Multiprocessing event to synchronize start.
                stop_event: Event to stop capture.
                tcp_event: Event for synchronization in TCP mode.
                is_tcp: Boolean indicating if operating in TCP mode.
                mp_stop_event: Optional event to wait before terminating process/thread.
                mh_num: Number of queue consumers (to send multiple `None` at the end).
                is_process: Boolean, True if running as a separate process.
                max_frames: Maximum desired FPS for capture.
            
            Raises:
                FileNotFoundError: If `video_path` does not exist.
                IOError: If the video cannot be opened.
            """

            if not os.path.exists(video_path):
                logging.error(f"Video file does not exist: {video_path}")
                for _ in range(mh_num): # Notify all consumers
                    frame_queue.put(None)
                raise FileNotFoundError(f"Video file does not exist: {video_path}")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logging.error(f"Error opening video file: {video_path}")
                for _ in range(mh_num): # Notify all consumers
                    frame_queue.put(None)
                raise IOError(f"Error opening video file: {video_path}")
            
            # Wait for TCP signal if enabled
            if is_tcp:
                tcp_event.wait()

            frame_count = 0
            first_time = True # To record the time of the first processed frame

            # Wait for t1_start signal
            t1_start.wait()
            
            # Calculate time per frame if max_fps is specified
            frame_time_target = 1 / max_frames if max_frames else None

            logging.info("Starting frame capture...")
            while cap.isOpened() and not stop_event.is_set():
                loop_start_time = time.time()
                
                if first_time: # This t1 seems to be for a benchmark, not for FPS logic
                    t1 = cv2.getTickCount() 
                    first_time = False
                    
                ret, frame = cap.read()

                if not ret:
                    logging.info("End of video or read error.")
                    break
                
                try:
                    # Try to put the frame in the queue, without waiting if there's an FPS limit
                    # (to discard frames if the queue is full and maintain the pace)
                    if max_frames:
                        frame_queue.put_nowait((frame, frame_count))
                    else:
                        frame_queue.put((frame, frame_count))
                
                except Exception as e: # It would be better to catch a more specific exception if known
                    logging.warning(f"Could not queue frame {frame_count}: {e}")
                    # Decide whether to continue or not, here the frame is simply skipped
                    pass
                    
                
                # If there's a frame_time_target, sleep to not exceed max_frames
                if frame_time_target:
                    elapsed_time = time.time() - loop_start_time
                    if elapsed_time < frame_time_target:
                        time.sleep(frame_time_target - elapsed_time)
                    
                frame_count += 1
                                               
            cap.release()
            logging.info(f"Capture finished. Total frames read: {frame_count}")

            # Signal the end to consumers
            for _ in range(mh_num):
                frame_queue.put(None)

            # Wait for the main process/thread stop signal if necessary
            if mp_stop_event:
                mp_stop_event.wait()
            
            # Terminate the process if running as such
            if is_process:
                logging.info("Terminating capture process.")
                os._exit(0)

    def process_frames(
        self,
        frame_queue: Union[mp.Queue, SharedCircularBuffer],
        detection_queue: Union[mp.Queue, SharedCircularBuffer],
        model_path: str,
        t1_start: mp.Event,
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        """Processes frames from a queue, performs object detection, and queues the results.

        Consumes frames from `frame_queue`, uses a YOLO model (loaded from
        `model_path`) to detect objects, and then queues the original frame
        along with detection results in `detection_queue`.
        Signals `t1_start` after initializing the model.

        Args:
            frame_queue: Input queue with frames to process.
            detection_queue: Output queue for frames with detections.
            model_path: Path to the YOLO model file.
            t1_start: Event to signal that model initialization has finished.
            mp_stop_event: Optional event to wait before terminating the process/thread.
            is_process: Boolean, True if running as a separate process.
        """
        from ultralytics import YOLO

        logging.info(f"Loading model from: {model_path}")
        model = YOLO(model_path, task="detect")

        # Model warm-up
        # This can improve the speed of the first real inferences.
        logging.info("Performing model warm-up...")
        # It's assumed that the model has these default parameters or they are configurable.
        # It's good practice to do warm-up with data similar to the input.
        # Here a generic configuration is used.
        try:
            model(conf=0.5, half=True, imgsz=(640, 640), augment=True, verbose=False) 
        except Exception as e:
            logging.warning(f"Error during model warm-up: {e}")
        
        logging.info("Model loaded and ready. Signaling t1_start.")
        t1_start.set() # Signal that the model is ready

        while True:
            item = frame_queue.get()
            if item is None: # End signal
                detection_queue.put(None) # Propagate the signal
                logging.info("End signal received in process_frames.")
                break

            frame, frame_count = item

            # Perform preprocessing (assuming model.predictor exists and has these methods)
            # It's important to verify the API of the ultralytics version being used.
            preprocessed_input = model.predictor.preprocess([frame])

            # Perform inference
            raw_output = model.predictor.inference(preprocessed_input) # 'inference' with 'n'

            # Postprocess the results
            results = model.predictor.postprocess(raw_output, preprocessed_input, [frame])

            # Format the result to be compatible with the tracker
            # It's assumed that the tracker expects a Namespace object with these fields.
            result_formatted = Namespace(
                xywh=results[0].boxes.xywh.cpu(), # Coordinates (center x, center y, width, height)
                conf=results[0].boxes.conf.cpu(), # Detection confidences
                cls=results[0].boxes.cls.cpu(),   # Detected classes
            )

            detection_queue.put((frame, result_formatted, frame_count))
        
        if mp_stop_event:
            mp_stop_event.wait()
        
        if is_process:
            logging.info("Terminating detection process.")
            os._exit(0)
    
    def tracking_frames(
        self,
        detection_queue: Union[mp.Queue, SharedCircularBuffer],
        tracking_queue: Union[mp.Queue, SharedCircularBuffer],
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        """Performs object tracking from detections.

        Consumes frames and detections from `detection_queue`, uses `TrackerWrapper`
        to perform object tracking, and queues the original frame
        along with tracked objects in `tracking_queue`.

        Args:
            detection_queue: Input queue with frames and detection results.
            tracking_queue: Output queue for frames with tracked objects.
            mp_stop_event: Optional event to wait before terminating the process/thread.
            is_process: Boolean, True if running as a separate process.
        """
        # Assumes TrackerWrapper is correctly implemented and configured.
        # The frame_rate could be dynamic or configured externally.
        tracker_wrapper = TrackerWrapper(frame_rate=30) 
        logging.info("Tracker initialized.")

        while True:
            item = detection_queue.get()
            if item is None: # End signal
                tracking_queue.put(None) # Propagate the signal
                logging.info("End signal received in tracking_frames.")
                break

            frame, result_detections, _ = item # frame_count is not used here directly

            # Perform tracking
            # `result_detections` must be compatible with what `tracker_wrapper.track` expects
            tracked_outputs = tracker_wrapper.track(result_detections, frame)

            tracking_queue.put((frame, tracked_outputs))
        
        if mp_stop_event:
            mp_stop_event.wait()
            
        if is_process:
            logging.info("Terminating tracking process.")
            os._exit(0)

    def tracking_frames_multihardware(
        self,
        detection_queue_GPU: Union[mp.Queue, SharedCircularBuffer],
        detection_queue_DLA0: Union[mp.Queue, SharedCircularBuffer],
        detection_queue_DLA1: Union[mp.Queue, SharedCircularBuffer],
        tracking_queue: Union[mp.Queue, SharedCircularBuffer],
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        """Performs object tracking from multiple detection queues (multi-hardware).

        Consumes frames and detections from three different queues (`detection_queue_GPU`,
        `detection_queue_DLA0`, `detection_queue_DLA1`), which are assumed to come from
        different hardware accelerators. Orders frames by their frame number
        before processing them with `TrackerWrapper` to maintain temporal coherence.
        Tracking results are queued in `tracking_queue`.

        Args:
            detection_queue_GPU: Detection queue for GPU.
            detection_queue_DLA0: Detection queue for DLA0.
            detection_queue_DLA1: Detection queue for DLA1.
            tracking_queue: Output queue for frames with tracked objects.
            mp_stop_event: Optional event to wait before terminating the process/thread.
            is_process: Boolean, True if running as a separate process.
        """
        tracker_wrapper = TrackerWrapper(frame_rate=30) # Adjust frame_rate if necessary
        logging.info("Multi-hardware tracker initialized.")

        # Flags to control if each input queue has finished
        stop_gpu, stop_dla0, stop_dla1 = False, False, False
        # Buffers to store the last item read from each queue
        item_gpu, item_dla0, item_dla1 = None, None, None

        while True:
            # Try to get a new item from each queue if it's not stopped and the buffer is empty
            if not stop_gpu and item_gpu is None:
                item_gpu = detection_queue_GPU.get()
                if item_gpu is None:
                    stop_gpu = True
                    logging.info("GPU queue finished.")

            if not stop_dla0 and item_dla0 is None:
                item_dla0 = detection_queue_DLA0.get()
                if item_dla0 is None:
                    stop_dla0 = True
                    logging.info("DLA0 queue finished.")

            if not stop_dla1 and item_dla1 is None:
                item_dla1 = detection_queue_DLA1.get()
                if item_dla1 is None:
                    stop_dla1 = True
                    logging.info("DLA1 queue finished.")

            # If all input queues have finished, terminate this process/thread
            if stop_gpu and stop_dla0 and stop_dla1:
                tracking_queue.put(None) # Signal the end to the next in the chain
                logging.info("All detection queues finished. Terminating multi-hardware tracking.")
                if mp_stop_event:
                    mp_stop_event.wait()
                if is_process:
                    os._exit(0)
                break # Exit the while loop

            # Extract frame numbers from current items (if they exist)
            # The expected item format is (frame, result, times, frame_number)
            # A very high value (float('inf')) is used if the item is None or doesn't have frame_number,
            # so that valid items have priority.
            frame_number_gpu = item_gpu[3] if item_gpu else float('inf')
            frame_number_dla0 = item_dla0[3] if item_dla0 else float('inf')
            frame_number_dla1 = item_dla1[3] if item_dla1 else float('inf')

            # Select the item with the lowest frame number to process
            # This ensures frames are processed in chronological order
            selected_item = None
            if frame_number_gpu <= frame_number_dla0 and frame_number_gpu <= frame_number_dla1 and item_gpu is not None:
                selected_item = item_gpu
                item_gpu = None # Empty the buffer so the next item from this queue is read
            elif frame_number_dla0 <= frame_number_gpu and frame_number_dla0 <= frame_number_dla1 and item_dla0 is not None:
                selected_item = item_dla0
                item_dla0 = None
            elif frame_number_dla1 <= frame_number_gpu and frame_number_dla1 <= frame_number_dla0 and item_dla1 is not None:
                selected_item = item_dla1
                item_dla1 = None
            else:
                # If there are no valid items or all buffers are empty (and some queue hasn't finished)
                # wait a bit to not consume CPU unnecessarily.
                # This can happen if one queue is much faster than the others and the others are waiting for data.
                if item_gpu is None and item_dla0 is None and item_dla1 is None and not (stop_gpu and stop_dla0 and stop_dla1):
                    time.sleep(0.001) # Small pause
                continue # Return to the beginning of the loop to re-evaluate or read new inputs

            frame, result_detections, _, _ = selected_item # times and frame_number are not used directly here

            # Perform tracking
            tracked_outputs = tracker_wrapper.track(result_detections, frame)
            tracking_queue.put((frame, tracked_outputs))

    def draw_and_write_frames(
        self,
        tracking_queue: Union[mp.Queue, SharedCircularBuffer],
        times_queue: Union[mp.Queue, SharedCircularBuffer], # Assume this queue is for benchmark times
        output_video_path: str,
        classes: dict, # Mapping of class ID to name
        memory: dict,  # Shared/updated tracking memory
        colors: dict,  # Mapping of class name to color for drawing
        stop_event: mp.Event, # Event to stop this process/thread
        tcp_conn_event: mp.Event, # Event to signal TCP connection establishment (renamed from tcp_conn)
        is_tcp: bool,
        mp_stop_event: Optional[mp.Event] = None,
        is_process: bool = False,
    ):
        """Draws tracked objects on frames, writes output video and handles TCP communication.

        Consumes frames with tracked objects from `tracking_queue`. Draws rectangles
        and labels for each object using information from `memory` and `colors`.
        Writes processed frames to a video file at `output_video_path`.
        If `is_tcp` is True, establishes a TCP server and sends "DETECTED_DEFECT"
        messages when a defective object is detected for the first time (according to `sended_id`).
        Signals `tcp_conn_event` after starting the TCP server.
        Upon completion, puts `None` in `times_queue` and activates `stop_event`.

        Args:
            tracking_queue: Input queue with frames and tracked objects.
            times_queue: Queue to send a completion signal (or times).
            output_video_path: Path to save the output video.
            classes: Dictionary mapping class ID to name. (Used indirectly via `update_memory`)
            memory: Tracking memory dictionary.
            colors: Dictionary mapping class name to color.
            stop_event: Global event to stop all processes/threads in the pipeline.
            tcp_conn_event: Event to signal that TCP connection is ready.
            is_tcp: Boolean indicating if operating in TCP mode.
            mp_stop_event: Optional event to wait before terminating the process/thread.
            is_process: Boolean, True if running as a separate process.
        """
        
        from concurrent.futures import ThreadPoolExecutor
        
        # ThreadPoolExecutor to handle background tasks (e.g. TCP sending)
        # max_workers could be adjusted according to expected load.
        thread_pool = ThreadPoolExecutor(max_workers=8) 
        video_writer = None # Initialize VideoWriter to None
        frame_number_counter = 0 # Counter for written frames
        
        # Dictionary to track defect IDs already sent via TCP
        # to avoid sending multiple messages for the same defect.
        sended_defect_ids = {} 

        client_socket, server_socket = None, None # Initialize sockets

        if is_tcp:
            try:
                logging.info("Starting TCP server on 0.0.0.0:8765...")
                client_socket, server_socket = tcp_server("0.0.0.0", 8765)
                # Send "READY" in a separate thread to not block.
                thread_pool.submit(handle_send, client_socket, "READY")
                tcp_conn_event.set() # Signal that TCP server is ready
            except Exception as e:
                raise RuntimeError(f"Error starting TCP server: {e}")


        while True:
            item = tracking_queue.get()
            if item is None: # End signal
                logging.info("End signal received in draw_and_write_frames.")
                break

            frame, tracked_objects = item

            # Initialize VideoWriter with the first frame to get dimensions
            if video_writer is None:
                try:
                    frame_height, frame_width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Codec for .mp4
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height)) # Fixed FPS at 30
                    logging.info(f"VideoWriter initialized for: {output_video_path}")
                except Exception as e:
                    logging.error(f"Error initializing VideoWriter: {e}")
                    break


            # Update memory with current tracked objects
            self.update_memory(tracked_objects, memory, classes)

            tcp_message_sent_this_frame = False

            # Internal function to draw a single object on the frame
            def draw_single_object(obj_data):
                nonlocal frame, memory, colors, tcp_message_sent_this_frame, is_tcp, client_socket, sended_defect_ids
                
                # Expected format: (xmin, ymin, xmax, ymax, obj_id, conf, ...)
                xmin, ymin, xmax, ymax, obj_id = map(int, obj_data[:5])
                confidence = float(obj_data[5])

                # Confidence threshold to draw the object
                if confidence < 0.4: 
                    return

                # Get updated class and memory state
                obj_id_in_memory = memory.get(obj_id)
                if not obj_id_in_memory:
                    return
                
                current_class_name = obj_id_in_memory["class"]
                is_currently_defective = current_class_name.endswith("-d")

                # TCP sending logic for defects
                if is_tcp and is_currently_defective and not tcp_message_sent_this_frame and not sended_defect_ids.get(obj_id):
                    sended_defect_ids[obj_id] = True

                    # Send TCP message in a pool thread to not block drawing
                    thread_pool.submit(handle_send, client_socket, "DETECTED_DEFECT")
                    tcp_message_sent_this_frame = True # Mark that a message was already sent in this frame
                    logging.debug(f"[TCP] Sending 'DETECTED_DEFECT' for ID {obj_id}")
                    
                # Draw rectangle and text
                object_color = colors.get(current_class_name, (255, 255, 255)) # Default color: white
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), object_color, 2)
                text_label = f"ID:{obj_id} {current_class_name} {confidence:.2f}"
                cv2.putText(
                   frame,
                   text_label,
                   (xmin, ymin - 10), # Text position above the rectangle
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, # Font size
                   (255, 255, 255), # Text color (white)
                   2, # Line thickness
                )

            # Draw all tracked objects using the thread pool
            draw_tasks = [thread_pool.submit(draw_single_object, obj) for obj in tracked_objects]
            for task in draw_tasks: # Wait for all drawing tasks to complete
                task.result()


            # Draw frame number on the video
            cv2.putText(
                frame,
                f"Frame: {frame_number_counter}",
                (10, 30), # Position
                cv2.FONT_HERSHEY_SIMPLEX,
                1, # Size
                (0, 255, 0), # Color (green)
                2, # Thickness
            )
            
            if video_writer:
                video_writer.write(frame)
            frame_number_counter += 1

        # Finalization and cleanup
        if video_writer:
            video_writer.release()
        
        thread_pool.shutdown(wait=True) # Close thread pool waiting for pending tasks to finish

     
        stop_event.set()      # Activate global stop event for other processes/threads

        if mp_stop_event:
            mp_stop_event.wait()

        if is_tcp and client_socket:
            try:
                client_socket.close()
            except Exception as e:
                print(f"Error closing TCP client socket: {e}")
        if is_tcp and server_socket:
            try:
                server_socket.close()
            except Exception as e:
                print(f"Error closing TCP server socket: {e}")


        if is_process:
            os._exit(0)

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
        mh_num: int = 1,
        is_process: bool = True,
    ):
        """Initializes the pipeline with common configuration and control events.

        Args:
            video_path: Path to the input video file.
            model_path: Path to the detection model file.
            output_video_path: Path to save the processed video.
            output_times: Path to save timing/benchmark information.
            parallel_mode: Parallelization mode (e.g. 'sequential', 'processes', 'threads').
            is_tcp: Enables TCP communication for defect notifications.
            sahi: Enables the use of SAHI (Slice-Aided Hyper Inference) for detection. (Not implemented in this fragment)
            max_fps: Limits the FPS of video capture.
            mh_num: Number of handlers/consumers for certain queues (multi-hardware/processing).
            is_process: Indicates if pipeline components run as separate processes.
        """
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.parallel_mode = parallel_mode
        self.is_tcp = is_tcp
        self.sahi = sahi
        self.max_fps = max_fps
        self.mh_num = mh_num
        self.is_process = is_process

        # Common control events
        self.tcp_event = mp.Event()
        self.stop_event = mp.Event()
        self.t1_start = mp.Event()
        self.mp_stop_event = mp.Event() if is_process else None

        # Shared memory
        self.memory = {}

    @abstractmethod
    def _initialize_queues(self):
        """Abstract method to initialize communication queues between stages.

        Must be implemented by derived classes to configure queues
        (e.g. `mp.Queue`, `SharedCircularBuffer`) according to the parallelization
        strategy and pipeline type.
        """
        pass

    @abstractmethod
    def _initialize_events(self):
        """Abstract method to initialize pipeline control events.

        Must be implemented by derived classes to configure events
        necessary for synchronization and pipeline flow control.
        """
        pass

    @abstractmethod
    def _create_workers(self):
        """Abstract method to create pipeline workers.
        Must be implemented by derived classes to start processes,
        threads, or any other parallel execution mechanism needed
        for pipeline stages.
        """
        pass

    @abstractmethod
    def _cleanup(self):
        """Abstract method for resource cleanup when finishing the pipeline.

        Must be implemented by derived classes to free resources such as
        processes, threads, queues, events, or any other handlers opened
        during pipeline execution.
        """
        pass

    @abstractmethod
    def run(self):
        """Executes the complete pipeline.

        This is the main method that orchestrates the startup, execution and
        orderly finalization of all pipeline stages.
        Must be implemented by derived classes.
        """
        pass
