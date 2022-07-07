import numbers

import cv2


class ResizeFlowNP:
    """Resize the np array and scale the value."""

    def __init__(self, size, scale_flow=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_flow = scale_flow

    def __call__(self, sample):

        th, tw = self.size
        h, w = sample.shape[0], sample.shape[1]
        sample = cv2.resize(sample, (tw, th), interpolation=cv2.INTER_LINEAR)
        if self.scale_flow:
            sample[:, :, 0] = sample[:, :, 0] * (float(tw) / float(w))
            sample[:, :, 1] = sample[:, :, 1] * (float(th) / float(h))

        return sample


class AverageMeter:
    """Computes and stores the average and current value Copied from:

    https://github.com/pytorch/examples/blob/master/imagenet/main.py.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
