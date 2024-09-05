from ultralytics import YOLO
import torch

def main():
    device = torch.device("cuda")

    model = YOLO('yolov10n.pt').to(device)

    model.train(data='data.yaml', epochs=100, batch=16, imgsz=640)

    model.val(data='data.yaml', batch=16)

    model = YOLO("/content/runs/detect/train4/weights/best.pt")
    model.export(format="tflite")

if __name__ == '__main__':
    main()