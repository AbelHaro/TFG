from detection_tracking_pipeline import DetectionTrackingPipeline
import torch.multiprocessing as mp  # type: ignore
import cv2


class DetectionTrackingPipelineWithMultiprocesses(DetectionTrackingPipeline):

    def __init__(
        self,
        video_path,
        model_path,
        output_video_path,
        output_times,
        parallel_mode,
        is_tcp=False,
        sahi=False,
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.output_times = output_times
        self.parallel_mode = parallel_mode
        self.is_tcp = is_tcp
        self.sahi = sahi

        # Colas específicas para multiprocesos
        self.frame_queue = mp.Queue(maxsize=10)
        self.detection_queue = mp.Queue(maxsize=10)
        self.tracking_queue = mp.Queue(maxsize=10)
        self.times_queue = mp.Queue(maxsize=10)

        # Memoria compartida
        self.memory = {}

        # Eventos de control
        self.tcp_event = mp.Event()
        self.stop_event = mp.Event()
        self.t1_start = mp.Event()
        self.mp_stop_event = mp.Event()

        # Parámetros adicionales
        self.is_process = True
        self.mh_num = 1

    def run(self):
        # Definir los procesos
        processes = [
            mp.Process(
                target=self.capture_frames,
                args=(
                    self.video_path,
                    self.frame_queue,
                    self.t1_start,
                    self.stop_event,
                    self.tcp_event,
                    self.is_tcp,
                    self.mp_stop_event,
                    self.mh_num,
                    self.is_process,
                ),
            ),
            mp.Process(
                target=self.process_frames_sahi if self.sahi else self.process_frames,
                args=(
                    self.frame_queue,
                    self.detection_queue,
                    self.model_path,
                    self.t1_start,
                    self.mp_stop_event,
                    self.is_process,
                ),
            ),
            mp.Process(
                target=self.tracking_frames,
                args=(
                    self.detection_queue,
                    self.tracking_queue,
                    self.mp_stop_event,
                    self.is_process,
                ),
            ),
            mp.Process(
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
                    self.is_process,
                ),
            ),
            mp.Process(
                target=self.write_to_csv,
                args=(
                    self.times_queue,
                    self.output_times,
                    self.parallel_mode,
                    self.t1_start,
                    self.stop_event,
                    self.mp_stop_event,
                    self.is_process,
                ),
            ),
            mp.Process(
                target=self.hardware_usage,
                args=(
                    self.parallel_mode,
                    self.stop_event,
                    self.t1_start,
                    self.tcp_event,
                    self.is_tcp,
                    self.is_process,
                ),
            ),
        ]

        # Iniciar todos los procesos
        t1 = cv2.getTickCount()
        for process in processes:
            process.start()

        # Esperar a que se detenga el pipeline
        self.stop_event.wait()

        # Calcular tiempo total y FPS
        t2 = cv2.getTickCount()
        time = (t2 - t1) / cv2.getTickFrequency()
        self.total_frames = self.get_total_frames(self.video_path)
        print(f"Total time: {time:.2f} s, FPS: {self.total_frames / time:.2f}")

        print("Pipeline finished.")
