import cv2
import json
import os

image_dir = "/Users/soles/Desktop/Radiographs_2k"
json_dir = '/Users/soles/Desktop/sorted_2k/'

for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)

    image = cv2.imread(image_path)
    mirrored_image = cv2.flip(image, 1)

    json_file = image_file[:-4] + ".json"
    json_path = os.path.join(json_dir, json_file)

    with open(json_path, "r") as f:
        json_data = json.load(f)
        for label in json_data['shapes']:
            bbox = label['points']
            [y1, x1], [y2, x2] = bbox
            mirrored_x1 = image.shape[1] - x2
            mirrored_x2 = image.shape[1] - x1
            mirrored_bbox = [[y1, mirrored_x1], [y2, mirrored_x2]]
            label['points'] = mirrored_bbox

        mirrored_image_path = os.path.join(image_dir, "mirrored_" + image_file)
        cv2.imwrite(mirrored_image_path, mirrored_image)

        mirrored_json_path = os.path.join(json_dir, "mirrored_" + json_file)
        with open(mirrored_json_path, "w") as f:
            json.dump(json_data, f, indent=4)
