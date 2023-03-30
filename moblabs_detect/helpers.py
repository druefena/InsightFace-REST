import numpy as np
from numba import njit
import logging


@njit(cache=True)
def nms(dets, thresh=0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



def resize_image(image, max_size: list = None):

    if max_size is None:
        max_size = [640, 640]

    cw = max_size[0]
    ch = max_size[1]
    h, w, _ = image.shape

    scale_factor = min(cw / w, ch / h)
    # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
    # so we reduce scale factor by some factor
    if scale_factor > 2:
        scale_factor = scale_factor * 0.7

    if scale_factor <= 1.:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    if scale_factor == 1.:
        transformed_image = image
    else:
        transformed_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                       fy=scale_factor,
                                       interpolation=interp)

    h, w, _ = transformed_image.shape

    if w < cw:
       transformed_image = cv2.copyMakeBorder(transformed_image, 0, 0, 0, cw - w,
                                                    cv2.BORDER_CONSTANT)
    if h < ch:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, ch - h, 0, 0,
                                                    cv2.BORDER_CONSTANT)

    return transformed_image, scale_factor