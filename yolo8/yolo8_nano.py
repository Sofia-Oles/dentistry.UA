from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# model = YOLO("yolov8n.yaml")

model.train(data="/Users/soles/Desktop/sorted_2k/YOLODataset/dataset.yaml",
            epochs=30,
            imgsz=640,
            workers=16,
            optimizer='Adam',
            lr0=0.0001)

metrics = model.val()
print(model.val().results_dict)
