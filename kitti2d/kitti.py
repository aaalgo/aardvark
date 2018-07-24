#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

class Object:
    def __init__ (self):
        pass

def load_label (path):
    objs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            obj = Object()
            obj.cat = line[0]
            obj.trunc = float(line[1])
            obj.occl = int(line[2])
            obj.alpha = float(line[3])
            obj.bbox = [float(x) for x in line[4:8]]
            obj.dim = [float(x) for x in line[8:11]]
            obj.loc = [float(x) for x in line[11:14]]
            obj.rot = float(line[14])
            objs.append(obj)
            pass
    return objs
