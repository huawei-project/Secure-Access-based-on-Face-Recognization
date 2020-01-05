# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 15:25:45
@LastEditTime : 2020-01-04 15:35:14
@Update: 
'''

import cv2
import numpy as np

sigmoid = lambda x: 1 / (1 + np.e**(-x))

def py_nms(dets, thresh, mode="Union"):
    """
    Params:
        dets:   {ndarray(n_boxes, 5)} x1, y1, x2, y2 score
        thresh: {float} retain overlap <= thresh
        mode:   {str} 'Union' or 'Minimum'
    Returns:
        keep:   {list[int]} indexes to keep
    Notes:
        greedily select boxes with high confidence
        keep boxes overlap <= thresh
        rule out overlap > thresh
    """
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
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def show_bbox(image, bbox, landmark=None, show_score=False):
    """
    Params: 
        image:  {ndarray(H, W, C)}
        bbox:   {ndarray(n_box, 5)} x1, y1, x2, y2, score
    """
    n_box = bbox.shape[0]
    for i_box in range(n_box):
        score = str(bbox[i_box, -1])
        x1, y1, x2, y2 = bbox[i_box, :-1].astype(np.int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255))
        if show_score:
            cv2.putText(image, str(score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if landmark is not None:
        for i_land in range(landmark.shape[0]):
            land = landmark[i_land].reshape((5, -1)).astype(np.int)
            for i_pt in range(land.shape[0]):
                cv2.circle(image, tuple(land[i_pt]), 1, (255, 255, 255), 2)

    cv2.imshow("", image)
    cv2.waitKey(0)

def _norm2(m):
    """
    Params:
        m: {ndarray(n, 2)}
    Returns:
        ret:{float} largest singular value
    Notes:
        求解矩阵2范数，即最大奇异值
    """
    u, s, vh = np.linalg.svd(m)
    ret = np.max(s)
    return ret

def _stitch(xy):
    """
    Params:
        xy:     {ndarray( n, 2)}
    Returns:
        ret:    {ndarray(2n, 4)}
    Notes：
        return
            x  y 1 0
            y -x 0 1
    """
    x, y = np.hsplit(xy, indices_or_sections=2)
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)

    ret = np.r_[np.c_[x,  y, ones, zeros], 
                np.c_[y, -x, zeros, ones]]

    return ret

def tformfwd(M, uv):
    """
    Params:
        M:  {ndarray(2, 3)}
        uv: {ndarray(n, 2)}
    Returns:
        ret: {ndarray(n, 2)}
    Notes:
        ret = [uv, 1] * M^T
    """
    ones = np.ones(shape=(uv.shape[0], 1))  # n x 2
    UV = np.c_[uv, ones]                    # n x 3
    ret = UV.dot(M.T)                       # n x 2
    return ret                              # n x 2

def findNonreflectiveSimilarity(uv, xy):
    """
    Params:
        uv: {ndarray(n, 2)}
        xy: {ndarray(n, 2)}
    Returns:
        M:  {ndarray(2, 3)}
    Notes:
        - Xr = U   ===>  r = (X^T X + \lambda I)^{-1} X^T U
        - r = [r1 r2 r3 r4]^T
        - M
            [r1 -r2 0
             r2  r1 0
             r3  r4 1]^{-1}[:, :2].T
    """
    X = _stitch(xy)
    U = uv.T.reshape(-1)
    r = np.linalg.pinv(X).dot(U)
    M = np.array(
        [[r[0], -r[1], 0],
         [r[1],  r[0], 0],
         [r[2],  r[3], 1]]
    )
    M = np.linalg.inv(M)
    return M[:, :2].T

def findReflectiveSimilarity(uv, xy):
    """
    Params:
        uv: {ndarray(n, 2)}
        xy: {ndarray(n, 2)}
    Returns:
        M:  {ndarray(2, 3)}
    """
    xyR = xy.copy(); xyR[:, 0] *= -1

    M1 = findNonreflectiveSimilarity(uv, xy)
    M2 = findNonreflectiveSimilarity(uv, xyR)
    
    M2[:, 0] *= -1

    xy1 = tformfwd(M1, uv)
    xy2 = tformfwd(M2, uv)

    norm1 = _norm2(xy1 - xy)
    norm2 = _norm2(xy2 - xy)

    return M1 if norm1 < norm2 else M2

def cp2tform(src, dst, mode = 'similarity'):
    """
    Params:
        src: {ndarray(n, 2)}
        dst: {ndarray(n, 2)}
        mode:{str} `similarity` or `noreflective`
    Returns:
        M:  {ndarray(2, 3)}
    """
    assert src.shape == dst.shape

    M = None
    if mode == 'similarity':
        M = findReflectiveSimilarity(src, dst)
    elif mode == 'noreflective':
        M = findNonreflectiveSimilarity(src, dst)
    else:
        print("Unsupported mode!")
    
    return M

def warpCoordinate(coord, M):
    """
    Params:
        coord: {ndarray(n, 2)}
        M:   {ndarray(2, 3)}
    """
    coord = np.c_[coord, np.ones(coord.shape[0])]
    coord = M.dot(coord.T).T
    return coord

def warpImage(im, M):
    return cv2.warpAffine(im, M, im.shape[:2][::-1])

def drawCoordinate(im, coord):
    """
    Params:
        im:  {ndarray(H, W, 3)}
        coord: {ndarray(n, 2)}
    Returns:
        im:  {ndarray(H, W, 3)}
    """
    for i in range(coord.shape[0]):
        cv2.circle(im, tuple(coord[i]), 1, (255, 255, 255), 3)
    return im

def drawCoordinate(im, coord):
    """
    Params:
        im:  {ndarray(H, W, 3)}
        coord: {ndarray(n, 2)}
    Returns:
        im:  {ndarray(H, W, 3)}
    """
    coord = coord.astype('int')
    for i in range(coord.shape[0]):
        cv2.circle(im, tuple(coord[i]), 1, (255, 255, 255), 3)
    return im
    
ALIGNED = [ 30.2946, 51.6963,
            65.5318, 51.5014,
            48.0252, 71.7366,
            33.5493, 92.3655,
            62.7299, 92.2041]

def imageAlignCrop(im, landmark, aligned=ALIGNED, dsize=(112, 96)):
    """
    Params:
        im:         {ndarray(H, W, 3)}
        landmark:   {ndarray(5, 2)}
        dsize:      {tuple/list(H, W)}
    Returns:
        dstImage:   {ndarray(h, w, 3)}
    Notes:
        对齐后裁剪
    """
    ## 变换矩阵
    M = cp2tform(landmark, np.array(aligned).reshape(-1, 2))
    
    ## 用矩阵变换图像
    warpedImage = warpImage(im, M)
    
    ## 裁剪固定大小的图片尺寸
    h, w = dsize
    dstImage = warpedImage[:h, :w]
    
    return dstImage