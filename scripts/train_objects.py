from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data='datasets/objects/data.yaml',
        epochs=100,
        imgsz=512,
        batch=16,
        patience=20,
        project='models',
        name='warden_objects',
        seed=42
    )

if __name__ == '__main__':
    main()