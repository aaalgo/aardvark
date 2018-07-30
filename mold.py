import numpy as np
import cv2
class Padding:
    def __init__ (self, stride):
        self.stride = stride
        pass

    def batch_image (self, image):
        # convert image into batch, with proper stride
        h, w = image.shape[:2]
        H = (h + self.stride - 1) // self.stride * self.stride
        W = (w + self.stride - 1) // self.stride * self.stride
        if len(image.shape) == 3:
            C = image.shape[2]
            batch = np.zeros((1, H, W, C), dtype=np.float32)
            batch[0, :h, :w, :] = image
        elif len(image.shape) == 2:
            batch = np.zeros((1, H, W, 1), dtype=np.float32)
            batch[0, :h, :w, 0] = image
        else:
            assert False
        return batch

    def unbatch_prob (self, image, prob_batch):
        # extract prob from a batch, image is only used for size
        h, w = image.shape[:2]
        assert prob_batch.shape[0] == 1
        return prob_batch[0, :h, :w]
    pass

class Scaling:
    def __init__ (self, stride, fixed = None):
        self.stride = stride
        self.fixed = None
        pass

    def batch_image (self, image):
        # convert image into batch, with proper stride
        h, w = image.shape[:2]
        H = (h + self.stride - 1) // self.stride * self.stride
        W = (w + self.stride - 1) // self.stride * self.stride
        if not self.fixed is None:
            H = self.fixed
            W = self.fixed
        if len(image.shape) == 3:
            C = image.shape[2]
            batch = np.zeros((1, H, W, C), dtype=np.float32)
            batch[0, :, :, :] = cv2.resize(image, (W, H))
        elif len(image.shape) == 2:
            batch = np.zeros((1, H, W, 1), dtype=np.float32)
            batch[0, :, :, 0] = cv2.resize(image, (W, H))
        else:
            assert False
        return batch

    def unbatch_prob (self, image, prob_batch):
        # extract prob from a batch, image is only used for size
        h, w = image.shape[:2]
        return cv2.resize(prob_batch[0], (w, h))
    pass
