#!/usr/bin/env python3
import os
import picpac

PARTS = {
'XR_ELBOW': 0,
'XR_FINGER': 1,
'XR_FOREARM': 2,
'XR_HAND': 3,
'XR_HUMERUS': 4,
'XR_SHOULDER': 5,
'XR_WRIST': 6
 }

def load_file (path):
    with open(path, 'rb') as f:
        return f.read()

def import_db (db_path, list_path):
    db = picpac.Writer(db_path, picpac.OVERWRITE)
    with open(list_path, 'r') as f:
        for l in f:
            path = l.strip()
            part = path.split('/')[2]
            #print(path)
            if 'positive' in path:
                l = 1
            elif 'negative' in path:
                l = 0
            else:
                assert 0

            pass
            assert part in PARTS
            k = PARTS[part]
            label = k * 2 + l
            db.append(label, load_file('data/' + path))
        pass
    pass

import_db('scratch/train.db', 'train.list')
import_db('scratch/val.db', 'val.list')
