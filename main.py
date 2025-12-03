import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

WEIGHTS_PATH = r"runs/detect/train/weights/best.pt"
VIDEO_SOURCE = r"sample3.mp4"
CONF_THRES = 0.4
OUTPUT_VIDEO = r"output_detections2.mp4"
DISPLAY_WINDOW = False
TRACK_HISTORY_LENGTH = 30

def calculate_speed(positions, fps):
    if len(positions) < 2:
        return 0
    
    p1 = positions[-2]
    p2 = positions[-1]
    distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    speed = distance * fps
    return speed

def main():
    model = YOLO(WEIGHTS_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error: could not open video source")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    print(f"Processing video: {VIDEO_SOURCE}")
    print(f"Output will be saved to: {OUTPUT_VIDEO}")
    print(f"FPS: {fps}")
    
    frame_count = 0
    display_enabled = DISPLAY_WINDOW
    track_history = defaultdict(lambda: [])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=CONF_THRES, persist=True)[0]

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            classes = results.boxes.cls.int().cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                track_history[track_id].append((center_x, center_y))
                
                if len(track_history[track_id]) > TRACK_HISTORY_LENGTH:
                    track_history[track_id].pop(0)
                
                speed = calculate_speed(track_history[track_id], fps)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                points = np.array(track_history[track_id], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(frame, [points], False, (255, 0, 255), 2)
                
                label = f"ID:{track_id} {model.names[cls_id]} {conf:.2f}"
                speed_label = f"Speed: {speed:.1f} px/s"
                
                cv2.putText(frame, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, speed_label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        out.write(frame)
        
        if display_enabled:
            try:
                cv2.imshow("Detections", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("Warning: Cannot display window. Continuing to save video...")
                display_enabled = False
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    if display_enabled:
        cv2.destroyAllWindows()
    
    print(f"\nDone! Processed {frame_count} frames.")
    print(f"Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
