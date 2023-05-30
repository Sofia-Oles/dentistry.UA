import cv2
from pathlib import Path
from ultralytics import YOLO

root = "C:\\Users\\soles\\Desktop\\git\\dentistryAI"
path = Path(f"{root}/runs/detect/train10/weights/best.pt")
model = YOLO(path)

test_path = '/Users/soles/Desktop/Radiographs_small/60.JPG'
test_image1 = cv2.imread(test_path)
results = model.predict([test_image1], save=True, line_thickness=1)
