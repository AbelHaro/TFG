import cv2
from ultralytics import YOLO
import os

def update_memory(track_id, detected_class, memory):
    """Update the memory with the state of detected objects."""
    if track_id not in memory:
        memory[track_id] = {'defective': detected_class.endswith('-d'), 'visible_frames': 30}
    else:
        memory[track_id]['defective'] |= detected_class.endswith('-d')
        memory[track_id]['visible_frames'] = 30  # Reset the frame counter

        if memory[track_id]['defective'] and not detected_class.endswith('-d'):
            detected_class = detected_class + '-d'

    memory[track_id]['class'] = detected_class

def process_frame(frame, model, classes, memory, total_times, colors):
    """Process a single frame and return the result."""
    t1 = cv2.getTickCount()
    results = model.track(source=frame, device=0, persist=True, tracker='bytetrack.yaml')
    t2 = cv2.getTickCount() 
    total_times["function_inference"] += (t2 - t1) / cv2.getTickFrequency()
    
    
    t1 = cv2.getTickCount()

    if results[0].boxes.id is None:
        t2 = cv2.getTickCount()
        total_times["process_frame"] += (t2 - t1) / cv2.getTickFrequency()
        
        return frame, results[0].speed, memory

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy().astype(float)

    for box, obj_id, cls, conf in zip(boxes, ids, detected_classes, confidences):
        xmin, ymin, xmax, ymax = box
        detected_class = classes[cls]
        
        update_memory(obj_id, detected_class, memory)
        
        detected_class = memory[obj_id]['class']
        color = colors.get(detected_class, (255, 255, 255))

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        text = f'ID:{obj_id} {detected_class} {conf:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        rect_x1 = xmin
        rect_y1 = ymin - 10 - text_height - baseline
        rect_x2 = xmin + text_width
        rect_y2 = ymin - 10
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        cv2.putText(frame, text, (xmin, ymin - 10), font, font_scale, (255, 255, 255), thickness)
        
    t2 = cv2.getTickCount()
    total_times["process_frame"] += (t2 - t1) / cv2.getTickFrequency()

    return frame, results[0].speed, memory

def process_video(video_path, model, output_video_path, classes, memory):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

    frame_count = 0
    total_times = {"preprocess": 0, "inference": 0, "postprocess": 0, "process_frame": 0, "write_frame": 0, "function_inference": 0}
    colors = {
        'negra': (0, 0, 255), 'blanca': (0, 255, 0), 'verde': (255, 0, 0), 'azul': (255, 255, 0),
        'negra-d': (0, 165, 255), 'blanca-d': (255, 165, 0), 'verde-d': (255, 105, 180), 'azul-d': (255, 0, 255)
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, speed, memory = process_frame(frame, model, classes, memory, total_times, colors)
        
        t1 = cv2.getTickCount()
        out.write(processed_frame)
        t2 = cv2.getTickCount()
        total_times["write_frame"] += (t2 - t1) / cv2.getTickFrequency()
        
        frame_count += 1
        total_times["preprocess"] += speed['preprocess']
        total_times["inference"] += speed['inference']
        total_times["postprocess"] += speed['postprocess']

        for track_id in list(memory):
            memory[track_id]['visible_frames'] -= 1
            if memory[track_id]['visible_frames'] <= 0:
                del memory[track_id]

    cap.release()
    out.release()
    return frame_count, total_times

def main():
    model_path = '../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.engine'
    video_path = '../datasets_labeled/videos/video_general_defectos_3.mp4'
    output_dir = '../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'video_con_tracking.mp4')

    model = YOLO(model_path)
    classes = {0: 'negra', 1: 'blanca', 2: 'verde', 3: 'azul', 4: 'negra-d', 5: 'blanca-d', 6: 'verde-d', 7: 'azul-d'}
    memory = {}

    total_start_time = cv2.getTickCount()
    frame_count, total_times = process_video(video_path, model, output_video_path, classes, memory)
    total_time = (cv2.getTickCount() - total_start_time) / cv2.getTickFrequency()

    print(f'Processed frames: {frame_count}')
    print(f'Total time: {total_time:.3f} seconds')
    print(f'Time per frame: {total_time / frame_count * 1000:.3f} ms')
    print(f'Detailed times (s): Preprocess: {total_times["preprocess"]/1000:.3f} s, Inference: {total_times["inference"]/1000:.3f} s, Postprocess: {total_times["postprocess"]/1000:.3f} s')
    print(f'Detailed times (s): Paint frame: {total_times["process_frame"]:.3f} s, Write frame: {total_times["write_frame"]:.3f} s')
    Tgpu = total_times["preprocess"] + total_times["inference"] + total_times["postprocess"]
    print(f'Total time GPU: {Tgpu/1000:.3f} s')
    print(f'Total time function inference: {total_times["function_inference"]:.3f} s')
    
    
if __name__ == "__main__":
    main()
