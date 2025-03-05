from detection_tracking_pipeline import DetectionTrackingPipeline
from queue import Queue
import torch.multiprocessing as mp # type: ignore
import threading

class DetectionTrackingPipelineWithThreads(DetectionTrackingPipeline):
    
    def __init__(self, video_path, model_path, output_video_path, tcp_conn=None, is_tcp=False):
        self.video_path = video_path
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.tcp_conn = tcp_conn
        self.is_tcp = is_tcp
        
        self.frame_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=10)
        self.tracking_queue = Queue(maxsize=10)
        self.times_queue = Queue(maxsize=10)
        
        self.memory = {}
        self.stop_event = mp.Event()
        
        self.t1_start = mp.Event()    
    
    def capture_frames(self, video_path, frame_queue, stop_event, tcp_conn, is_tcp):
        return super().capture_frames(video_path, frame_queue, stop_event, tcp_conn, is_tcp)
    
    def process_frames(self, frame_queue, detection_queue, model_path, t1_start):
        return super().process_frames(frame_queue, detection_queue, model_path, t1_start)
    
    def tracking_frames(self, detection_queue, tracking_queue):
        return super().tracking_frames(detection_queue, tracking_queue)
    
    def draw_and_write_frames(self, tracking_queue, times_queue, output_video_path, classes, memory, colors, stop_event, tcp_conn, is_tcp):
        return super().draw_and_write_frames(tracking_queue, times_queue, output_video_path, classes, memory, colors, stop_event, tcp_conn, is_tcp)
    
    def write_to_csv(self, times_queue, output_file):
        return super().write_to_csv(times_queue, output_file)
    
    def hardware_usage(self, output_file, stop_event, t1_start, tcp_conn, is_tcp):
        return super().hardware_usage(output_file, stop_event, t1_start, tcp_conn, is_tcp)
    
    def run(self):
        threads = [
            threading.Thread(target=self.capture_frames, args=(self.video_path, self.frame_queue, self.stop_event, self.tcp_conn, self.is_tcp)),
            threading.Thread(target=self.process_frames, args=(self.frame_queue, self.detection_queue, self.model_path, self.t1_start)),
            threading.Thread(target=self.tracking_frames, args=(self.detection_queue, self.tracking_queue)),
            threading.Thread(target=self.draw_and_write_frames, args=(self.tracking_queue, self.times_queue, self.output_video_path, self.CLASSES, self.memory, self.COLORS, self.stop_event, self.tcp_conn, self.is_tcp)),
            threading.Thread(target=self.write_to_csv, args=(self.times_queue, self.output_video_path)),
            threading.Thread(target=self.hardware_usage, args=(self.output_video_path, self.stop_event, self.t1_start, self.tcp_conn, self.is_tcp)),
        ]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
            
        print("[DETECTION_TRACKING_PIPELINE_WITH_THREADS] Finished running threads.")

            
        
    