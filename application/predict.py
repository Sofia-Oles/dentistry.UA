import os
import shutil
from pathlib import Path
from ultralytics import YOLO

root = "C:\\Users\\soles\\Desktop\\git\\dentistryAI"
path = Path('C:\\Users\\soles\\Desktop\\git\\dentistryAI\\runs\\detect\\train10\\weights\\best.pt')


def predict_image(image, image_name):
    model = YOLO(path)
    model.predict([image], save=True, line_thickness=1)

    # Rename image0.jpg file from default models root path /runs/detect/predict
    output_dir = Path(f"{root}/runs/detect/predict")
    old_name = os.path.join(output_dir, "image0.jpg")
    new_name = os.path.join(output_dir, f"{image_name}")
    os.rename(old_name, new_name)

    new_folder = Path(f"{root}/application/static/uploads")
    new_path = os.path.join(new_folder, f"{image_name}")
    shutil.move(new_name, new_path)
