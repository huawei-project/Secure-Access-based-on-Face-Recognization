# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 11:25:12
@LastEditTime : 2020-01-05 16:51:57
@Update: 
'''
import os
import cv2
import numpy as np

import torch
import torch.cuda as cuda
import torch.nn as nn
from torchvision.transforms import ToTensor

from utils import sigmoid, py_nms, show_bbox

class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 10, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 16, 3, 1)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(32, 15, 1, 1)    # 1 + 4 + 10

    def forward(self, x):
        """
        Params:
            x: {tensor(N, 3, H, W)}
        Returns:
            x: {tensor(N, 15, H // 12, W // 12)}
        """
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.prelu3(x)
        
        x = self.conv4(x)

        return x


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 28, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(28, 48, 3, 1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(48, 64, 2, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Linear(64*2*2, 128)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Linear(128,  15)    # 1 + 4 + 10

    def forward(self, x):
        """
        Params:
            x: {tensor(N, 3, 24, 24)}
        Returns:
            x: {tensor(N, 15)}
        """

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.prelu3(x)

        x = x.view(x.shape[0], -1)
        
        x = self.conv4(x)
        x = self.prelu4(x)

        x = self.conv5(x)

        return x


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.conv1 = nn.Conv2d( 3,  32, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(32,  64, 3, 1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(64,  64, 3, 1)
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 2, 1)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Linear(256,  15)    # 1 + 4 + 10

    def forward(self, x):
        """
        Params:
            x: {tensor(N, 3, 48, 48)}
        Returns:
            x: {tensor(N, 15)}
        """
        
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.prelu4(x)

        x = x.view(x.shape[0], -1)
        
        x = self.conv5(x)
        x = self.prelu5(x)

        x = self.conv6(x)

        return x
        

class MtcnnDetector(object):
    """ mtcnn detector

    Params:
        prefix: {str} checkpoint
    Attributes:

    Content:

    """
    def __init__(self, min_face=20, thresh=[0.6, 0.7, 0.7], scale=0.79, stride=2, cellsize=12, use_cuda=False, ckpt='../ckpt'):
        
        self.min_face = min_face
        self.thresh = thresh
        self.scale  = scale
        self.stride = stride
        self.cellsize = cellsize
        self.use_cuda = use_cuda
        
        self.pnet, self.rnet, self.onet = self._load(ckpt)
    
    def _load(self, ckptdir):

        pnet, rnet, onet = PNet(), RNet(), ONet()
        for net in [pnet, rnet, onet]:
            ckpt = torch.load(
                os.path.join(ckptdir, '{}.pkl'.format(net._get_name())), map_location='cpu')
            net.load_state_dict(ckpt['net_state'])
            if cuda.is_available() and self.use_cuda:
                net.cuda()
            net.eval()
        return pnet, rnet, onet
    
    def detect_image(self, image):
        """ Detect face over single image
        Params:
            image:    {ndarray(H, W, C)}
        """

        boxes, boxes_c, landmark = self._detect_pnet(image)
        boxes, boxes_c, landmark = self._detect_rnet(image, boxes_c)
        boxes, boxes_c, landmark = self._detect_onet(image, boxes_c)
        return boxes_c, landmark

    def _detect_pnet(self, image):
        """
        Params:
            image:      {ndarray(1, C, H, W)}
        Returns:
            boxes:    {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            boxes_c:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            landmark: None
        """
        NETSIZE = 12

        def _resize_image(image, scale):
            """ resize image according to scale
            Params:
                image:  {ndarray(h, w, c)}
                scale:  {float}
            """
            h, w, c = image.shape
            hn = int(h*scale); wn = int(w*scale)
            resized = cv2.resize(image, (wn, hn), interpolation=cv2.INTER_LINEAR)
            return resized
        
        def _generate_box(cls_map, reg_map, thresh, scale):
            """ generate boxes
            Params:
                cls_map: {ndarray(h, w)}
                reg_map: {ndarray(4, h, w)}
                thresh:  {float}
                scale:   {float}
            Returns:
                bboxes:  {ndarray(n_boxes, 9)} x1, y1, x2, y2, score, offsetx1, offsety1, offsetx2, offsety2
            """
            idx = np.where(cls_map>thresh)

            if idx[0].size == 0:
                return np.array([])

            x1 = np.round(self.stride * idx[1] / scale)
            y1 = np.round(self.stride * idx[0] / scale)
            x2 = np.round((self.stride * idx[1] + self.cellsize) / scale)
            y2 = np.round((self.stride * idx[0] + self.cellsize) / scale)

            # print("current scale: {} current size: {}".format(scale, self.cellsize/scale))

            score = cls_map[idx[0], idx[1]]
            reg = np.array([reg_map[i, idx[0], idx[1]] for i in range(4)])

            boxes = np.vstack([x1, y1, x2, y2 ,score, reg]).T

            return boxes

        # ======================= generate boxes ===========================
        cur_scale = NETSIZE / self.min_face
        cur_img = _resize_image(image, cur_scale)
        all_boxes = None

        while min(cur_img.shape[:-1]) >= NETSIZE:

            ## forward network
            X = ToTensor()(cur_img).unsqueeze(0)
            if cuda.is_available() and self.use_cuda: X = X.cuda()
            with torch.no_grad():
                y_pred = self.pnet(X)[0].cpu().detach().numpy()

            ## generate bbox
            cls_map = sigmoid(y_pred[0,:,:])
            reg_map = y_pred[1:5,:,:]
            boxes = _generate_box(cls_map, reg_map, self.thresh[0], cur_scale)

            ## update scale
            cur_scale *= self.scale
            cur_img = _resize_image(image, cur_scale)
            if boxes.size == 0: continue
            
            ## nms
            # boxes = boxes[self._nms(boxes[:, :5], 0.6, 'Union')]

            ## save bbox
            if all_boxes is None:
                all_boxes = boxes
            else:
                all_boxes = np.concatenate([all_boxes, boxes], axis=0)

        # ====================================================================

        if all_boxes is None: 
            return np.array([]), np.array([]), None

        ## nms
        all_boxes = all_boxes[self._nms(all_boxes[:, 0:5], 0.6, 'Union')]

        ## parse
        boxes  = all_boxes[:, :4]                   # (n_boxes, 4)
        score  = all_boxes[:,  4].reshape((-1, 1))  # (n_boxes, 1)
        offset = all_boxes[:, 5:]                   # (n_boxes, 4)
        
        # refine bbox
        boxes_c = self._cal_box(boxes, offset)
        
        ## concat
        boxes = np.concatenate([boxes, score], axis=1)
        boxes_c = np.concatenate([boxes_c, score], axis=1)

        ## landmark
        landmark = None

        return boxes, boxes_c, landmark

    def _detect_rnet(self, image, bboxes):
        """
        Params:
            image: {ndarray(H, W, C)}
            bboxes:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
        Returns:
            boxes:    {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            boxes_c:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            landmark: None
        """
        NETSIZE = 24

        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), None

        bboxes = self._square(bboxes)
        patches = self._crop_patch(image, bboxes, NETSIZE)
        
        ## forward network
        X = torch.cat(list(map(lambda x: ToTensor()(x).unsqueeze(0), patches)), dim=0)
        if cuda.is_available() and self.use_cuda: X = X.cuda()
        with torch.no_grad():
            y_pred = self.rnet(X).cpu().detach().numpy()  # (n_boxes, 15)
        scores = sigmoid(y_pred[:, 0])          # (n_boxes,)
        offset = y_pred[:, 1: 5]                # (n_boxes, 4)
        landmark = y_pred[:, 5:]                # (n_boxes, 10)

        ## update score
        bboxes[:, -1] = scores

        ## filter
        idx = scores > self.thresh[1]
        bboxes = bboxes[idx]                        # (n_boxes, 5)
        offset = offset[idx]                        # (n_boxes, 4)
        landmark = landmark[idx]                    # (n_boxes, 10)
        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), None

        ## nms
        idx = self._nms(bboxes, 0.5)
        bboxes = bboxes[idx]
        offset = offset[idx]
        landmark = landmark[idx]

        ## landmark
        landmark = self._cal_landmark(bboxes[:, :-1], landmark)

        bboxes_c = self._cal_box(bboxes[:,:-1], offset)
        bboxes_c = np.concatenate([bboxes_c, bboxes[:, -1].reshape((-1, 1))], axis=1)

        return bboxes, bboxes_c, landmark
    
    def _detect_onet(self, image, bboxes):
        """
        Params:
            image: {ndarray(H, W, C)}
            bboxes:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
        Returns:
            boxes:    {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            boxes_c:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            landmark: {ndarray(n_boxes, 10)} x1, y1, x2, y2, ..., x5, y5
        """
        NETSIZE = 48

        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        bboxes = self._square(bboxes)
        patches = self._crop_patch(image, bboxes, NETSIZE)
        
        ## forward network
        X = torch.cat(list(map(lambda x: ToTensor()(x).unsqueeze(0), patches)), dim=0)
        if cuda.is_available() and self.use_cuda: X = X.cuda()
        with torch.no_grad():
            y_pred = self.onet(X).cpu().detach().numpy()  # (n_boxes, 15)
        scores = sigmoid(y_pred[:, 0])          # (n_boxes,)
        offset = y_pred[:, 1: 5]                # (n_boxes, 4)
        landmark = y_pred[:, 5:]                # (n_boxes, 10)
        
        ## update score
        bboxes[:, -1] = scores

        ## filter
        idx = scores > self.thresh[2]
        bboxes = bboxes[idx]                        # (n_boxes, 5)
        offset = offset[idx]                        # (n_boxes, 4)
        landmark = landmark[idx]                    # (n_boxes, 10)
        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        ## nms
        idx = self._nms(bboxes, 0.5, mode='Minimum')
        bboxes = bboxes[idx]
        offset = offset[idx]
        landmark = landmark[idx]
        
        ## landmark
        landmark = self._cal_landmark(bboxes[:, :-1], landmark)

        bboxes_c = self._cal_box(bboxes[:,:-1], offset)
        bboxes_c = np.concatenate([bboxes_c, bboxes[:, -1].reshape((-1, 1))], axis=1)

        return bboxes, bboxes_c, landmark

    @classmethod
    def _cal_box(self, boxes, offset):
        """ refine boxes
        Params:
            boxes:  {ndarray(n_boxes, 4)} unrefined boxes
            offset: {ndarray(n_boxes, 4)} boxes offset
        Returns:
            boxes_c:{ndarray(n_boxes, 4)} refined boxes
        Notes:
            offset = (gt - square) / size of square box
             => gt = square + offset * size of square box (*)
            where
                - `offset`, `gt`, `square` are ndarrays
                - `size of square box` is a number
        """
        ## square boxes' heights and widths
        x1, y1, x2, y2 = np.hsplit(boxes, 4)        # (n_boxes, 1)
        w = x2 - x1 + 1; h = y2 - y1 + 1            # (n_boxes, 1)
        bsize = np.hstack([w, h]*2)                 # (n_boxes, 4)
        bbase = np.hstack([x1, y1, x2, y2])         # (n_boxes, 4)
        ## refine
        boxes_c = bbase + offset*bsize
        return boxes_c
    
    @classmethod
    def _cal_landmark(self, boxes, offset):
        """ calculate landmark
        Params:
            boxes:  {ndarray(n_boxes,  4)} unrefined boxes
            offset: {ndarray(n_boxes, 10)} landmark offset
        Returns:
            landmark:{ndarray(n_boxes, 10)} landmark location
        Notes:
            offset_x = (gt_x - square_x1) / size of square box
             => gt_x = square_x1 + offset_x * size of square box (*)
            offset_y = (gt_y - square_y1) / size of square box
             => gt_y = square_y1 + offset_y * size of square box (*)
            where
                - `offset_{}`, `gt_{}`, `square_{}1` are ndarrays
                - `size of square box` is a number
        """
        ## square boxes' heights and widths
        x1, y1, x2, y2 = np.hsplit(boxes, 4)        # (n_boxes, 1)
        w = x2 - x1 +1; h = y2 - y1 + 1             # (n_boxes, 1)
        bsize = np.hstack([w, h]*5)                 # (n_boxes, 10)
        bbase = np.hstack([x1, y1]*5)               # (n_boxes, 10)
        ## refine
        landmark = bbase + offset*bsize
        return landmark

    @classmethod
    def _nms(self, dets, thresh, mode="Union"):
        """
        Params:
            dets:   {ndarray(n_boxes, 5)} x1, y1, x2, y2 score
            thresh: {float} retain overlap <= thresh
            mode:   {str} 'Union' or 'Minimum'
        Returns:
            idx:   {list[int]} indexes to keep
        Notes:
            greedily select boxes with high confidence
            idx boxes overlap <= thresh
            rule out overlap > thresh

            if thresh==1.0, keep all
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        idx = []
        while order.size > 0:
            i = order[0]
            idx.append(i)

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

        return idx
    
    @classmethod
    def _square(self, bbox):
        """ convert rectangle bbox to square bbox
        Params:
            bbox: {ndarray(n_boxes, 5)}
        Returns:
            bbox_s: {ndarray(n_boxes, 5)}
        """
        ## rectangle boxes' heights and widths
        x1, y1, x2, y2, score = np.hsplit(bbox, 5)  # (n_boxes, 1)
        w = x2 - x1 +1; h = y2 - y1 + 1             # (n_boxes, 1)
        maxsize = np.maximum(w, h)                  # (n_boxes, 1)

        ## square boxes' heights and widths
        x1 = x1 + w/2 - maxsize/2
        y1 = y1 + h/2 - maxsize/2
        x2 = x1 + maxsize - 1
        y2 = y1 + maxsize - 1

        bbox_s = np.hstack([x1, y1, x2, y2, score])
        return bbox_s

    @classmethod
    def _crop_patch(self, image, bbox_s, size):
        """ crop patches from image
        Params:
            image: {ndarray(H, W, C)}
            bbox_s: {ndarray(n_boxes, 5)} squared bbox
        Returns:
            patches: {list[ndarray(h, w, c)]}
        """

        def locate(bbox, imh, imw):
            """ 
            Params:
                bbox:       {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
                imh, imw:   {float} size of input image
            Returns:
                oriloc, dstloc: {ndarray(n_boxes, 4)} x1, y1, x2, y2
            """
            ## origin boxes' heights and widths
            x1, y1, x2, y2, score = np.hsplit(bbox_s, 5)# (n_boxes, 1)
            x1, y1, x2, y2 = list(map(lambda x: x.astype('int').reshape(-1), [x1, y1, x2, y2]))
            w = x2 - x1 + 1; h = y2 - y1 + 1            # (n_boxes, 1)

            ## destinate boxes
            xx1 = np.zeros_like(x1)
            yy1 = np.zeros_like(y1)
            xx2 = w.copy() - 1
            yy2 = h.copy() - 1

            ## left side out of image
            i = x1 < 0
            xx1[i] = 0 + (0 - x1[i])
            x1 [i] = 0
            ## top side out of image
            i = y1 < 0
            yy1[i] = 0 + (0 - y1[i])
            y1 [i] = 0
            ## right side out of image
            i = x2 > imw - 1
            xx2[i] = (w[i]-1) + (imw-1 - x2[i])
            x2 [i] = imw - 1
            ## bottom side out of image
            i = y2 > imh - 1
            yy2[i] = (h[i]-1) + (imh-1 - y2[i])
            y2 [i] = imh - 1

            return [x1, y1, x2, y2, xx1, yy1, xx2, yy2]

        imh, imw, _ = image.shape

        x1, y1, x2, y2, score = np.hsplit(bbox_s, 5)    
        pw = x2 - x1 + 1; ph = y2 - y1 + 1
        pshape = np.hstack([ph, pw, 3*np.ones(shape=(score.shape[0], 1))]).astype('int')   # (n_boxes, 3)
        # keep = np.bitwise_or(pw > 0, ph > 0).reshape(-1)
        # pshape = pshape[keep]; bbox_s = bbox_s[keep]
        n_boxes = bbox_s.shape[0]

        x1, y1, x2, y2, xx1, yy1, xx2, yy2 = locate(bbox_s, imh, imw) # (n_boxes, 1)

        patches = []
        for i_boxes in range(n_boxes):
            patch = np.zeros(shape=pshape[i_boxes], dtype='uint8')
            patch[yy1[i_boxes]: yy2[i_boxes], xx1[i_boxes]: xx2[i_boxes]] = \
                        image[y1[i_boxes]: y2[i_boxes], x1[i_boxes]: x2[i_boxes]]
            patch = cv2.resize(patch, (size, size))
            patches += [patch]
        
        return patches

def test():

    import sys
    
    detector = MtcnnDetector(min_face=12, thresh=[0.9, 0.8, 0.7], scale=0.79, stride=2, cellsize=12, use_cuda=False)
    
    imgfile = sys.argv[1]

    image = cv2.imread(imgfile)
    
    boxes, boxes_c, landmark = detector._detect_pnet(image)
    show_bbox(image.copy(), boxes, None, False)
    show_bbox(image.copy(), boxes_c, landmark, False)

    boxes, boxes_c, landmark = detector._detect_rnet(image, boxes_c)
    show_bbox(image.copy(), boxes, None, False)
    show_bbox(image.copy(), boxes_c, landmark, False)

    boxes, boxes_c, landmark = detector._detect_onet(image, boxes_c)
    show_bbox(image.copy(), boxes, None, False)
    show_bbox(image.copy(), boxes_c, landmark, True)

# if __name__ == "__main__":

    # test()