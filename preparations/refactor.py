import os
import json

categories = ['orthpedic crown', 'orphotedic crown', 'crown', 'orthopedic crown']
right_name = 'orthopedic crown'

directory = "/Users/soles/Desktop/sorted_3"

for filename in os.listdir(directory):
    if filename.endswith(".json"):

        with open(os.path.join(directory, filename), "r") as f:
            data = json.load(f)

        for i in data['shapes']:
            if i['label'] in categories:
                i['label'] = right_name

        with open(os.path.join(directory, filename), "w") as f:
            json.dump(data, f)
