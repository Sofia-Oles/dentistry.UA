import json
import os
import cv2
from PIL import Image
from preparations.utils import get_code

directory = "/Users/soles/Desktop/Radiographs_2k"


for filename in os.listdir(directory):
    if filename.endswith(".JPG"):
        file = directory + "/" + filename
        rotated_file = directory + "/rotated_" + filename

        raw_img = cv2.imread(file)
        (h, w) = raw_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated_img = cv2.warpAffine(raw_img, M, (w, h))
        cv2.imwrite(rotated_file, rotated_img)

        img = Image.open(rotated_file)
        json_file = '/Users/soles/Desktop/sorted_2k/' + filename[:-4] + '.json'
        rotated_json_file = '/Users/soles/Desktop/sorted_2k/rotated_' + filename[:-4] + '.json'
        with open(json_file, 'r') as f:
            file = json.load(f)

        for label in file['shapes']:
            bbox = label['points']
            [y1, x1], [y2, x2] = bbox
            x1, y1 = h - x1, w - y1
            x2, y2 = h - x2, w - y2
            label['points'] = [[y2, x2], [y1, x1]]

        img_data, h, w = get_code(img)
        file['imageData'] = img_data

        with open(rotated_json_file, 'w') as f:
            json.dump(file, f)
