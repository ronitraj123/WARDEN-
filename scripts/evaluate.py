from ultralytics import YOLO

def main():
    model = YOLO('models/warden_objects/weights/best.pt')
    metrics = model.val(
        data='datasets/objects/data.yaml',
        split='test',      # <-- key difference: run on test, not val
        conf=0.4,
        iou=0.5
    )
    print(f"Overall mAP50: {metrics.box.map50:.3f}")
    print(f"Overall mAP50-95: {metrics.box.map:.3f}")
    print(f"Per-class mAP50-95: {metrics.box.maps}")

if __name__ == '__main__':
    main()