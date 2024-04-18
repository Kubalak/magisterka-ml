from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    model.train(data="dataset_config.yaml", epochs=10, imgsz=416, name="lego")