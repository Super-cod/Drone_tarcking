from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(data="C:/Users/swaya/Desktop/Timepass/inside_fpv/Drone_tarcking/Birds&Drons-1/data.yaml",
                epochs=40,
                imgsz=640,
                batch=8,  # Reduced batch size for 4GB GPU
                patience=10,  
                save=True,
                plots=True)
