import cv2
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

WEIGHTS_PATH = r"weights\best.pt"
VIDEO_SOURCE = r"..\yolo\sample4.mp4"
OUTPUT_VIDEO = r"output_classification_tracking.mp4"
CONF_THRES = 0.5
DISPLAY_WINDOW = False
TRACK_HISTORY_LENGTH = 30
CLASS_NAMES = ['bird', 'drone']

def load_model(weights_path, num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def calculate_speed(positions, fps):
    if len(positions) < 2:
        return 0
    
    p1 = positions[-2]
    p2 = positions[-1]
    distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    speed = distance * fps
    return speed

def classify_region(model, frame, box, device, transform):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0, 0.0
    
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def main():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        print("Please train the model first by running train.py")
        return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(WEIGHTS_PATH, num_classes=len(CLASS_NAMES))
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"\nModel loaded from: {WEIGHTS_PATH}")
    print(f"Classes: {CLASS_NAMES}\n")
    
    # Use YOLOv8 for detection and tracking
    from ultralytics import YOLO
    detector = YOLO(r"..\yolo\runs\detect\train4\weights\best.pt")
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {VIDEO_SOURCE}")
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
        
        # Detect and track objects with YOLO
        results = detector.track(frame, persist=True, verbose=False)[0]
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            for box, track_id, det_conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Classify the detected region
                class_id, cls_conf = classify_region(model, frame, (x1, y1, x2, y2), device, transform)
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                track_history[track_id].append((center_x, center_y))
                
                if len(track_history[track_id]) > TRACK_HISTORY_LENGTH:
                    track_history[track_id].pop(0)
                
                speed = calculate_speed(track_history[track_id], fps)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center dot
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Draw trajectory
                points = np.array(track_history[track_id], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(frame, [points], False, (255, 0, 255), 2)
                
                # Labels
                label = f"ID:{track_id} {CLASS_NAMES[class_id]} {cls_conf:.2f}"
                speed_label = f"Speed: {speed:.1f} px/s"
                
                cv2.putText(frame, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, speed_label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        out.write(frame)
        
        if display_enabled:
            try:
                cv2.imshow("Classification Tracking", frame)
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
