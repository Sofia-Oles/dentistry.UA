import torch
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
# model = YOLO("yolov8n.yaml")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.train(data="/Users/soles/Desktop/sorted_2k/YOLODataset/dataset.yaml",
            epochs=30,
            batch=10,
            imgsz=640,
            workers=16,
            optimizer='Adam',
            lr0=0.0001,
            device=device)

metrics = model.val()
print(model.val().results_dict)
