from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train/weights/best.pt")
    model.train(data="C:/Users/swaya/Desktop/Timepass/inside_fpv/Drone_tarcking/Birds&Drons-1/data.yaml",
                epochs=80,
                imgsz=640,
                batch=8,
                patience=15,
                save=True,
                plots=True,
                resume=False,
                lr0=0.0001,
                warmup_epochs=1,
                device=0)
