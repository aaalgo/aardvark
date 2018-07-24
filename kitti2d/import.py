#!/usr/bin/env python3
import picpac 
from tqdm import tqdm
import simplejson as json
from kitti import *

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
json.encoder.c_make_encoder = None


def read_list (path):
    nums = []  # read bname list
    with open(path, 'r') as f:
        for l in f:
            nums.append(int(l.strip()))
            pass
        pass
    return nums

def load_file (path):
    with open(path, 'rb') as f:
        return f.read()

def import_db (db_path, list_path):
    db = picpac.Writer(db_path, picpac.OVERWRITE)

    tasks = read_list(list_path)
    #for number in tqdm(tasks):
    for number in tasks:
        image_path = os.path.join('data/training/image_2', '%06d.png' % number)
        label_path = os.path.join('data/training/label_2', '%06d.txt' % number)

        image = cv2.imread(image_path, -1)
        label = load_label(label_path)

        H, W = image.shape[:2]

        shapes = []
        for obj in label:
            if obj.cat != 'Car':
                continue
            #print(obj.bbox)
            x1, y1, x2, y2 = obj.bbox
            x = x1 / W
            y = y1 / H
            w = (x2 - x1)/ W
            h = (y2 - y1)/ H
            shapes.append({'type': 'rect', 'geometry': {'x': x, 'y': y, 'width': w, 'height': h}})
        anno = {'shapes': shapes, 'number': number}
        anno_buf = json.dumps(anno).encode('ascii')
        #print(anno_buf)
        db.append(0, load_file(image_path), anno_buf)
    pass

import_db('scratch/train.db', 'train.txt')
