#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import StratifiedKFold
import cv2
from chest import *
import picpac

def load_file (path):
    with open(path, 'rb') as f:
        return f.read()

def import_db (path, tasks):
    with open(path + '.list','w') as f:
        db = picpac.Writer(path, picpac.OVERWRITE)
        for p, l in tqdm(list(tasks)):
            f.write('%s,%d\n' % (p, l))
            image = cv2.imread(p, -1)
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            image_buffer = cv2.imencode('.jpg', image)[1].tostring()
            db.append(float(l), image_buffer)
            pass

X = []
Y = []
with open('data/Data_Entry_2017.csv', 'r') as f:
    f.readline()
    for l in f:
        bname, labels, _ = l.strip().split(',', 2)
        labels = [x.strip() for x in labels.split('|')]
        if len(labels) != 1:
            continue
        label = labels[0]
        l = LABEL_LOOKUP.get(label, -1)
        path = image_path(bname)
        if path is None or l == -1:
            continue
        X.append(path)
        Y.append(l)
        #db.append(l, load_file(path))
print('Found %d images.' % len(X))

X = np.array(X)
Y = np.array(Y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

for train_index, val_index in skf.split(np.zeros(len(X)), Y):
    import_db('scratch/train.db', zip(X[train_index], Y[train_index]))
    import_db('scratch/val.db', zip(X[val_index], Y[val_index]))
    break

