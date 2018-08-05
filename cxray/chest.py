#!/usr/bin/env python3

import os
import sys
from tqdm import tqdm
from glob import glob
import pickle

CATEGORIES = [
	[60361, 'No Finding'],
	[11559, 'Atelectasis'],
	#[1, 'bels'],
	[2776, 'Cardiomegaly'],
	[4667, 'Consolidation'],
	[2303, 'Edema'],
	[13317, 'Effusion'],
	[2516, 'Emphysema'],
	[1686, 'Fibrosis'],
	#[227, 'Hernia'],
	[19894, 'Infiltration'],
	[5782, 'Mass'],
	[6331, 'Nodule'],
	[3385, 'Pleural_Thickening'],
	[1431, 'Pneumonia'],
	[5302, 'Pneumothorax'],
]

CLASSES = len(CATEGORIES)
LABEL_LOOKUP = {}
for l, (_, v) in enumerate(CATEGORIES):
    LABEL_LOOKUP[v] = l
    pass

LOOKUP_PATH = 'lookup.pickle'
if os.path.exists(LOOKUP_PATH):
    with open(LOOKUP_PATH, 'rb') as f:
        lookup = pickle.load(f)
else:
    lookup = {}
    print("Scanning images...")
    for i in range(1, 13):
        C = 0
        for p in glob('data/images_%03d/*.png' % i):
            bname = os.path.basename(p)
            #print(bname)
            lookup[bname] = i
            C += 1
        print('%d found for directory %d' % (C, i))
        pass

    with open(LOOKUP_PATH, 'wb') as f:
        pickle.dump(lookup, f)
    pass

def image_path (bname):
    n = lookup.get(bname, None)
    if n is None:
        return n
    return 'data/images_%03d/%s' % (lookup[bname], bname)



