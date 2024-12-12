import cv2
from ultralytics import YOLO
import os
import multiprocessing
from multiprocessing import Process, Manager, Value
from ultralytics.trackers.byte_tracker import BYTETracker
from argparse import Namespace
import time
import numpy as np
from create_excel import create_excel

FRAME_AGE = 15

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def update_memory(tracked_objects, memory, classes):
    global FRAME_AGE
    
    for obj in tracked_objects:
        track_id = int(obj[4])
        detected_class = classes[int(obj[6])]
        
        is_defective = detected_class.endswith('-d')
        if track_id in memory:
            entry = memory[track_id]
            entry['defective'] |= is_defective
            entry['visible_frames'] = FRAME_AGE
            if entry['defective'] and not is_defective:
                detected_class += '-d'
            entry['class'] = detected_class
        else:
            memory[track_id] = {'defective': is_defective, 'visible_frames': FRAME_AGE, 'class': detected_class}
            
    for track_id in list(memory):
        memory[track_id]['visible_frames'] -= 1
        if memory[track_id]['visible_frames'] <= 0:
            del memory[track_id]

def capture_frames(video_path, pipe_send):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Error al abrir el archivo de video: {video_path}")
    
    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        t2 = cv2.getTickCount()

        if not ret:
            break
        pipe_send.send(frame)
    
    cap.release()
    pipe_send.send(None)

def process_frames(pipe_recv, pipe_send, model_path):
    model = YOLO(model_path, task='detect')
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy_frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
    
    while True:
        frame = pipe_recv.recv()
        if frame is None:
            pipe_send.send(None)
            break
        t1 = cv2.getTickCount()
        results = model.predict(source=frame, device=0, conf=0.2, imgsz=(640, 640), half=True, augment=True, task='detect')
        t2 = cv2.getTickCount()


        pipe_send.send((frame, results[0].boxes.cpu()))
        

class TrackerWrapper:
    global FRAME_AGE
    def __init__(self, frame_rate=20):
        self.args = Namespace(
            tracker_type='bytetrack',
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=FRAME_AGE,
            match_thresh=0.8,
            fuse_score=True
        )
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)
    
    class Detections:
        def __init__(self, boxes, confidences, class_ids):
            self.conf = confidences
            self.xywh = boxes
            self.cls = class_ids
    
    def track(self, detection_data, frame):
        detections = self.Detections(
            detection_data.xywh.cpu().numpy(),
            detection_data.conf.cpu().numpy(),
            detection_data.cls.cpu().numpy().astype(int)
        )
        return self.tracker.update(detections, frame)

def tracking_frames(pipe_recv, pipe_send):
    tracker_wrapper = TrackerWrapper(frame_rate=20)
    
    while True:
        item = pipe_recv.recv()
        if item is None:
            pipe_send.send(None)
            break

        t1 = cv2.getTickCount()
        frame, result = item
        outputs = tracker_wrapper.track(result, frame)
        t2 = cv2.getTickCount()

        pipe_send.send((frame, outputs))
    

def draw_and_write_frames(pipe_recv, output_video_path, classes, memory, colors):
    frame_number = 0
    out = None
    
    while True:
        item = pipe_recv.recv()
        if item is None:
            break
        t1 = cv2.getTickCount()
        frame, tracked_objects = item

        if out is None:
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

        update_memory(tracked_objects, memory, classes)
        
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])
            conf = float(obj[5])
            
            if conf < 0.4:
                continue
            
            detected_class = memory[obj_id]['class']
            color = colors.get(detected_class, (255, 255, 255))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            text = f'ID:{obj_id} {detected_class} {conf:.2f}'
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        frame_number += 1
        cv2.imshow('frame', frame)
        out.write(frame)
        
        t2 = cv2.getTickCount()

    if out:
        out.release()
        
    cv2.destroyAllWindows()

def main():
    model_path = '../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16.engine'
    video_path = '../datasets_labeled/videos/contar_canicas.mp4'
    output_dir = '../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'multiprocesos.mp4')

    classes = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    colors = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }

    memory = {}
    

    t1 = cv2.getTickCount()
    with multiprocessing.Pool(processes=4) as pool:
        parent_conn1, child_conn1 = multiprocessing.Pipe()
        parent_conn2, child_conn2 = multiprocessing.Pipe()
        parent_conn3, child_conn3 = multiprocessing.Pipe()

        pool.apply_async(capture_frames, args=(video_path, child_conn1))
        pool.apply_async(process_frames, args=(parent_conn1, child_conn2, model_path))
        pool.apply_async(tracking_frames, args=(parent_conn2, child_conn3))
        pool.apply_async(draw_and_write_frames, args=(parent_conn3, output_video_path, classes, memory, colors))

        pool.close()
        pool.join()
    t2 = cv2.getTickCount()
    
    total_time = (t2 - t1) / cv2.getTickFrequency()
    total_frames = get_total_frames(video_path)
    
    print(f"Tiempo total: {total_time:.2f} segundos")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')   
    print("Number of cpu : ", multiprocessing.cpu_count())
    main()
