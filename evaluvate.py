from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train/weights/best.pt")

    results = model.val(
        data="disability.yaml",
        split="val"
    )

    print("\n📊 Evaluation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

    if hasattr(results.box, "ap_class_index"):
        for i, cls_name in enumerate(results.names.values()):
            print(f"\nClass: {cls_name}")
            print(f"  AP50: {results.box.ap50[i]:.4f}")
            print(f"  AP50-95: {results.box.ap[i]:.4f}")

if __name__ == "__main__":
    main()