#!/usr/bin/env python3 

import os 
import picpac 
from glob import glob

dic = { 'nogate.txt':0,
        'gateclose.txt':1,
        'gatefar.txt':2}

def load_file(path):
    with open(path, 'rb') as f:
        return f.read()

def load_anno(path):
    anno_dic = {}
    for files in os.listdir(path):
        #print (files)
        for line in open(os.path.join(path,files)):
            line = line.strip('\n')
            anno_dic[line]=dic[files]
            #print (line,dic[files])
    return anno_dic

anno = load_anno("/shared/s2/users/wdong/football/annotation")

def import_db (db_path, img_path):
    db = picpac.Writer(db_path, picpac.OVERWRITE)
    for image in glob(img_path+"*/*.jpg"):
        file_name = image.split("/")[-1]
        if file_name in anno.keys():
            db.append(anno[file_name],load_file(image),image.encode('ascii'))

    pass

#import_db('./scratch/train.db','/shared/s2/users/wdong/football/trainingthumbs/')
import_db('./scratch/val.db','/shared/s2/users/wdong/football/testthumbs/')


