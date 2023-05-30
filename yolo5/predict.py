import cv2
from pathlib import Path
from ultralytics import YOLO

root = "C:\\Users\\soles\\Desktop\\git\\dentistryAI"
path = Path(f"{root}/runs/detect/train7/weights/best.pt")
model = YOLO(path)

# work
test_path = '/Users/soles/Desktop/Radiographs_2k/163.JPG'
test_image1 = cv2.imread(test_path)
# cv2.imshow('Image', test_image1)
# cv2.waitKey(0)
results = model.predict([test_image1], save=True, line_thickness=1)
