from detection_tracking_pipeline import DetectionTrackingPipeline
import torch.multiprocessing as mp  # type: ignore
import cv2
from classes.shared_circular_buffer import SharedCircularBuffer


class DetectionTrackingPipelineWithMultiprocessesSharedMemory(DetectionTrackingPipeline):

    def __init__(
        self,
        video_path,
        model_path,
        output_video_path,
        output_times,
        parallel_mode,
        is_tcp=False,
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.parallel_mode = parallel_mode
        self.is_tcp = is_tcp

        self.frame_queue = SharedCircularBuffer(queue_size=10, max_item_size=16)
        self.detection_queue = SharedCircularBuffer(queue_size=10, max_item_size=4)
        self.tracking_queue = SharedCircularBuffer(queue_size=10, max_item_size=4)
        self.times_queue = SharedCircularBuffer(queue_size=10, max_item_size=4)

        self.memory = {}

        self.tcp_event = mp.Event()

        self.stop_event = mp.Event()

        self.t1_start = mp.Event()

        self.mp_stop_event = mp.Event()

    def capture_frames(self, video_path, frame_queue, stop_event, tcp_event, is_tcp, mp_stop_event):
        return super().capture_frames(
            video_path, frame_queue, stop_event, tcp_event, is_tcp, mp_stop_event=mp_stop_event
        )

    def process_frames(self, frame_queue, detection_queue, model_path, t1_start, mp_stop_event):
        return super().process_frames_sahi(
            frame_queue, detection_queue, model_path, t1_start, mp_stop_event=mp_stop_event
        )

    def tracking_frames(self, detection_queue, tracking_queue, mp_stop_event):
        return super().tracking_frames(detection_queue, tracking_queue, mp_stop_event=mp_stop_event)

    def draw_and_write_frames(
        self,
        tracking_queue,
        times_queue,
        output_video_path,
        classes,
        memory,
        colors,
        stop_event,
        tcp_event,
        is_tcp,
        mp_stop_event,
    ):
        return super().draw_and_write_frames(
            tracking_queue,
            times_queue,
            output_video_path,
            classes,
            memory,
            colors,
            stop_event,
            tcp_event,
            is_tcp,
            mp_stop_event=mp_stop_event,
        )

    def write_to_csv(self, times_queue, output_file, parallel_mode, stop_event, mp_stop_event):
        return super().write_to_csv(
            times_queue, output_file, parallel_mode, stop_event, mp_stop_event=mp_stop_event
        )

    def hardware_usage(self, parallel_mode, stop_event, t1_start, tcp_event, is_tcp):
        return super().hardware_usage(parallel_mode, stop_event, t1_start, tcp_event, is_tcp)

    def run(self):
        processes = [
            mp.multiprocessing.Process(
                target=self.capture_frames,
                args=(
                    self.video_path,
                    self.frame_queue,
                    self.stop_event,
                    self.tcp_event,
                    self.is_tcp,
                    self.mp_stop_event,
                ),
            ),
            mp.multiprocessing.Process(
                target=self.process_frames,
                args=(
                    self.frame_queue,
                    self.detection_queue,
                    self.model_path,
                    self.t1_start,
                    self.mp_stop_event,
                ),
            ),
            mp.multiprocessing.Process(
                target=self.tracking_frames,
                args=(self.detection_queue, self.tracking_queue, self.mp_stop_event),
            ),
            mp.multiprocessing.Process(
                target=self.draw_and_write_frames,
                args=(
                    self.tracking_queue,
                    self.times_queue,
                    self.output_video_path,
                    self.CLASSES,
                    self.memory,
                    self.COLORS,
                    self.stop_event,
                    self.tcp_event,
                    self.is_tcp,
                    self.mp_stop_event,
                ),
            ),
            mp.multiprocessing.Process(
                target=self.write_to_csv,
                args=(
                    self.times_queue,
                    self.output_times,
                    self.parallel_mode,
                    self.stop_event,
                    self.mp_stop_event,
                ),
            ),
            mp.multiprocessing.Process(
                target=self.hardware_usage,
                args=(
                    self.parallel_mode,
                    self.stop_event,
                    self.t1_start,
                    self.tcp_event,
                    self.is_tcp,
                ),
            ),
        ]

        t1 = cv2.getTickCount()

        for process in processes:
            process.start()

        self.stop_event.wait()

        t2 = cv2.getTickCount()
        time = (t2 - t1) / cv2.getTickFrequency()

        self.frame_queue.close()
        self.detection_queue.close()
        self.tracking_queue.close()
        self.times_queue.close()

        self.frame_queue.unlink()
        self.detection_queue.unlink()
        self.tracking_queue.unlink()
        self.times_queue.unlink()

        self.total_frames = self.get_total_frames(self.video_path)

        print(
            f"[DETECTION_TRACKING_PIPELINE_WITH_THREADS] Total time: {time:.2f} s, FPS: {self.total_frames / time:.2f}"
        )

        print("[DETECTION_TRACKING_PIPELINE_WITH_THREADS] Finished running threads.")
