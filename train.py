from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  

    results = model.train(
        data="disability.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        device="cpu"
    )

    model.val()
    model.export(format="onnx")
if __name__ == "__main__":
    main()
