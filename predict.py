import os
import cv2
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train/weights/best.pt")

    test_dir = "test/images"
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(test_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_dir, file)
            results = model(img_path)

            for r in results:
                im = r.plot()  
                save_path = os.path.join(output_dir, file)
                cv2.imwrite(save_path, im)

            print(f"Processed {file} → saved to {output_dir}/")

if __name__ == "__main__":
    main()
