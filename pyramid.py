#!/usr/bin/env python3
import numpy as np
import cv2

def pyramid_helper (canvas, mask, rois, canvas_offset, image, horizontal, threshold):
    H, W = canvas.shape[:2]
    h, w = image.shape[:2]
    if min(h, w) < threshold:
        return

    x0, y0 = canvas_offset


    canvas[:h, :w, :] = image
    mask[:h, :w] = len(rois)
    rois.append([x0, y0, w, h])


    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    h2, w2 = image.shape[:2]

    if horizontal:
        o = W-w2
        canvas = canvas[:,o:,:]
        mask = mask[:,o:]
        x0 += o
    else:
        o = H-h2
        canvas = canvas[o:, :, :]
        mask = mask[o:, :]
        y0 += o
    pyramid_helper(canvas, mask, rois, (x0, y0), image, not horizontal, threshold)
    pass

class Pyramid:
    def __init__ (self, image, threshold=64, stride=16, min_size=600):
        self.image = image
        # returns canvas, mask, rois
        # canvas, the image sprial
        # mask: the depth of each pixel
        # rois[depth] = (x, y, W, H) 
        h, w = image.shape[:2]
        m = min(h, w)
        if m < min_size:
            ratio = min_size / m
            image = cv2.resize(image, None, fx=ratio, fy=ratio)
            h, w = image.shape[:2]

        C = 1
        if len(image.shape) == 3:
            C = image.shape[2]
        #else:
        #    image = np.reshape(a, (H, W, 1))

        H = (h + stride -1) // stride * stride
        W = (w * 2 + stride - 1) // stride * stride

        canvas = np.zeros((H, W, C), image.dtype)
        mask = np.zeros((H, W), np.int32)
        rois = [(0, 0, 0, 0)]   # rois[0] is not used
        pyramid_helper(canvas, mask, rois, (0, 0), image, True, threshold)
        if C == 1:
            canvas = canvas[:, :, 0]
        self.pyramid = canvas
        self.mask = mask
        self.rois = rois
        pass

    def find_depth (self, box):
        x1, y1, x2, y2 = np.round(box).astype(np.int32)
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        roi = self.mask[y1:(y2+1), x1:(x2+1)]
        uniq, cnts = np.unique(roi, return_counts=True)
        return uniq[np.argmax(cnts)]

    def combine (self, boxes):
        R = []
        h, w = self.image.shape[:2]
        for box in boxes:
            d = self.find_depth(box)
            if d == 0:
                continue
            x0, y0, w0, h0 = self.rois[d]
            x1, y1, x2, y2 = box
            x1 = (x1 - x0) * w / w0
            x2 = (x2 - x0) * w / w0
            y1 = (y1 - y0) * h / h0
            y2 = (y2 - y0) * h / h0
            R.append([x1, y1, x2, y2])
            pass
        return R

if __name__ == '__main__':
    image = cv2.imread('lenna.png', -1)
    sp = Pyramid(image, 16)
    cv2.imwrite('pyramid.png', sp.pyramid)
    cv2.normalize(sp.mask, sp.mask, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('pyramid_mask.png', sp.mask)
    pass

