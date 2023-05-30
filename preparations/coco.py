import os
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)

transform = transforms.Compose([
    transforms.Resize((640, 1615)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = '/Users/soles/Desktop/Radiographs_rotated'
json_path = '/Users/soles/Desktop/jsons_rotated'
annotation_path = '/Users/soles/Desktop/file.json'

coco_annotations = {
    'info': {},
    'licenses': [],
    'categories': [],
    'images': [],
    'annotations': []
}

categories = [
    {'id': 1, 'name': 'implant', 'supercategory': 'object'},
    {'id': 2, 'name': 'canal e-sealing', 'supercategory': 'object'},
    {'id': 3, 'name': 'e-sealing', 'supercategory': 'object'},
    {'id': 4, 'name': 'facet', 'supercategory': 'object'},
    {'id': 5, 'name': 'filling', 'supercategory': 'object'},
    {'id': 6, 'name': 'orthopedic crown', 'supercategory': 'object'},
    {'id': 7, 'name': 'healthy', 'supercategory': 'object'},
    {'id': 8, 'name': 'decay', 'supercategory': 'object'}
]

for category in categories:
    coco_annotations['categories'].append(category)

image_id = 0
annotation_id = 0

for filename in os.listdir(json_path):
    if filename.endswith('.json'):
        with open(os.path.join(json_path, filename), 'r') as f:
            json_data = json.load(f)

        image_filename = os.path.join(data_path, json_data['imagePath'])
        image = Image.open(image_filename)

        image_info = {
            'id': image_id,
            'file_name': json_data['imagePath'],
            'height': image.height,
            'width': image.width
        }
        coco_annotations['images'].append(image_info)

        for obj in json_data['shapes']:
            bbox = obj['points']
            bbox = np.array(bbox).reshape(-1)
            xmin = min(bbox[0::2])
            ymin = min(bbox[1::2])
            xmax = max(bbox[0::2])
            ymax = max(bbox[1::2])

            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': [item['id'] for item in categories if item["name"] == obj['label']][0],
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'area': (xmax - xmin) * (ymax - ymin),
                'iscrowd': 0
            }
            coco_annotations['annotations'].append(annotation)
            annotation_id += 1
        image_id += 1

# print(coco_annotations)
my_dict_str = json.dumps(coco_annotations, cls=NumpyArrayEncoder)

with open('my_dict.json', 'w') as file:
    file.write(my_dict_str)
