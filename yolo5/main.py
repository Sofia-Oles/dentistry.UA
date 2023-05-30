from ultralytics import YOLO

model = YOLO("yolov5s.pt")
# model = YOLO('yolo5.yaml')

model.train(data="/Users/soles/Desktop/jsons_rotated/YOLODataset/dataset.yaml",
            epochs=20,
            imgsz=640,
            workers=16,
            optimizer='Adam')

metrics = model.val()

# 7 train - 20 epoch
# {'metrics/precision(B)': 0.7976204431375261,
#  'metrics/recall(B)': 0.8171483362199934,
#  'metrics/mAP50(B)': 0.818137021886784,
#  'metrics/mAP50-95(B)': 0.554785338308852,
#  'fitness': 0.5811205066666453}
