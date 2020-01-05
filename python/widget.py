# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 17:09:26
@LastEditTime : 2020-01-05 17:04:06
@Update: 
'''
import os
import cv2
import numpy as np

import torch

from PyQt5.QtWidgets import QDialog, QDesktopWidget
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import QTimer

from mtcnn import MtcnnDetector
from mobilefacenet import MobileFacenet

class _Dialog(QDialog):

    def __init__(self, parent=None):
        super(_Dialog, self).__init__(parent)
    
        self._params = dict()
        self._qLineEdits = dict()
        
        self._file  = '../param/{}.txt'.format(self.__class__.__name__)
        if os.path.exists(self._file): self._load()
        
        self._setupUi()

    # ------------------------- UI界面 -------------------------
    def _setupUi(self):
        """ 其他 """
        self._baseUi()  # 基本
        self._setup()   # 其余，待实现
        self._show()    # 显示
    
    def _baseUi(self):
        """ 基本按钮 """
        # ---------- 主窗口属性 ----------
        self.setGeometry(0, 0, 400, 300)
        s, f = QDesktopWidget().screenGeometry(), self.geometry()
        self.move((s.width() - f.width()) / 2, (s.height() - f.height()) / 2)

        # ------------ 按键 -------------
        self.pbSave = QPushButton('Save', self)
        self.pbSave.setGeometry(220, 250, 70, 25)
        self.pbSave.clicked[bool].connect(self._onPushButtonSave_clicked)
        
        self.pbApply = QPushButton('Apply', self)
        self.pbApply.setGeometry(300, 250, 70, 25)
        self.pbApply.clicked[bool].connect(self._onPushButtonApply_clicked)

    def _setup(self):

        raise NotImplementedError

    def _show(self):
        for k, v in self._params.items():
            self._qLineEdits[k].setText(str(v))

    # ------------------------- 参数读写 -------------------------
    def _save(self):

        s = '\n'.join([' '.join([k, str(v)]) for k, v in self._params.items()])
        with open(self._file, 'w') as f:
            f.write(s)

    def _load(self):

        with open(self._file, 'r') as f:
            s = f.read()
        self._params = {k: float(v) for k, v in [l.split(' ') for l in s.split('\n')]}

    # -------------------------- slots --------------------------
    def _onPushButtonSave_clicked(self):

        self._save()

    def _onPushButtonApply_clicked(self):

        """ 应用参数，重新初始化 """
        if self.isOpened():
            self.release()
            self.open()
        self.close()

    def open():
        raise NotImplementedError

    def isOpened():
        raise NotImplementedError

    def release():
        raise NotImplementedError


class WidgetCamera(_Dialog):

    _names = ['Index', 'FPS', 'Scale']

    def __init__(self, parent=None):
        super(WidgetCamera, self).__init__(parent)

        self._capture = None

    def _setup(self):

        self.setWindowTitle("Camera")

        onLineEdits = dict(zip(self._names,[
            self._onLineEditIndex_textChanged, 
            self._onLineEditFps_textChanged, 
            self._onLineEditScale_textChanged, 
        ]))

        x1, y1, w, h, gw, gh = 110, 80, 100, 25, 10, 5

        for i, name in enumerate(self._names):
            lb = QLabel(self);    lb.setGeometry(x1,          y1 + (h + gh) * i, w, h); lb.setText(name)
            le = QLineEdit(self); le.setGeometry(x1 + w + gw, y1 + (h + gh) * i, w, h)
            le.textChanged[str].connect(onLineEdits[name])
            self._qLineEdits[name] = le

    # ----------------------- slots -----------------------
    def _onLineEditIndex_textChanged (self, text): self.index  = float(text)
    def _onLineEditFps_textChanged   (self, text): self.fps    = float(text)
    def _onLineEditScale_textChanged (self, text): self.scale  = float(text)

    # ----------------- public functions ----------------
    def open(self):
        """ 打开相机 """
        self._capture = cv2.VideoCapture(self.index)
        return self.isOpened()
    
    def isOpened(self):
        """ 相机是否正常使用 """
        if self._capture is None:
            return False
        return self._capture.isOpened()
    
    def release(self):
        """ 关闭相机 """
        if self.isOpened():
            self._capture.release()
            self._capture = None

    def __call__(self):
        """ 读取图片 """
        if self.isOpened():
            return self._capture.read()
        else:
            return False, None

    # -------------------- property --------------------
    @property
    def index(self):
        return int(self._params['Index'])
    
    @index.setter
    def index(self, value):
        self._params['Index'] = value

    @property
    def fps(self):
        return self._params['FPS']
    
    @fps.setter
    def fps(self, value):
        self._params['FPS'] = value
    
    @property
    def scale(self):
        return self._params['Scale']
    
    @scale.setter
    def scale(self, value):
        self._params['Scale'] = value


class WidgetDetector(_Dialog):

    _names = ['Minface', 'P', 'R', 'O']

    def __init__(self, parent=None):
        super(WidgetDetector, self).__init__(parent)

        self.open()

    def _setup(self):

        self.setWindowTitle("Detector")

        onLineEdits = dict(zip(self._names,[
            self._onLineEditMinface_textChanged, 
            self._onLineEditP_textChanged, 
            self._onLineEditR_textChanged, 
            self._onLineEditO_textChanged, 
        ]))

        x1, y1, w, h, gw, gh = 110, 80, 100, 25, 10, 5

        for i, name in enumerate(self._names):
            lb = QLabel(self);    lb.setGeometry(x1,          y1 + (h + gh) * i, w, h); lb.setText(name)
            le = QLineEdit(self); le.setGeometry(x1 + w + gw, y1 + (h + gh) * i, w, h)
            le.textChanged[str].connect(onLineEdits[name])
            self._qLineEdits[name] = le

    # ----------------------- slots -----------------------
    def _onLineEditMinface_textChanged(self, text): self.minface = float(text)
    def _onLineEditP_textChanged(self, text):       self.threshp = float(text)
    def _onLineEditR_textChanged(self, text):       self.threshr = float(text)
    def _onLineEditO_textChanged(self, text):       self.thresho = float(text)

    # ----------------- public functions ----------------
    def open(self):
        self._detector = MtcnnDetector(
            min_face=self.minface, 
            thresh=[self.threshp, self.threshr, self.thresho]
        )
    
    def isOpened(self):
        self._detector is not None
    
    def release(self):
        self._detector = None

    def __call__(self, image):
        """
        Params:

        Returns:
            bbox:     {ndarray(n_boxes,  5)} x1, y1, x2, y2, score
            landmark: {ndarray(n_boxes, 10)} x1, y1, x2, y2, ..., x5, y5
        """
        return self._detector.detect_image(image)

    # -------------------- property --------------------
    @property
    def minface(self):
        return self._params['Minface']
    
    @minface.setter
    def minface(self, value):
        self._params['Minface'] = value

    @property
    def threshp(self):
        return self._params['P']
    
    @threshp.setter
    def threshp(self, value):
        self._params['P'] = value

    @property
    def threshr(self):
        return self._params['R']
    
    @threshr.setter
    def threshr(self, value):
        self._params['R'] = value

    @property
    def thresho(self):
        return self._params['O']
    
    @thresho.setter
    def thresho(self, value):
        self._params['O'] = value


class WidgetVerifier(_Dialog):

    _names = ['Thresh', 'N(save)', 'N(pass)']
    _ndim  = 256

    def __init__(self, parent=None):
        super(WidgetVerifier, self).__init__(parent)

        self.open()

    def _setup(self):

        self.setWindowTitle("Verifier")

        onLineEdits = dict(zip(self._names,[
            self._onLineEditThresh_textChanged,
            self._onLineEditNsave_textChanged,
            self._onLineEditNpass_textChanged,
        ]))

        x1, y1, w, h, gw, gh = 110, 80, 100, 25, 10, 5

        for i, name in enumerate(self._names):
            lb = QLabel(self);    lb.setGeometry(x1,          y1 + (h + gh) * i, w, h); lb.setText(name)
            le = QLineEdit(self); le.setGeometry(x1 + w + gw, y1 + (h + gh) * i, w, h)
            le.textChanged[str].connect(onLineEdits[name])
            self._qLineEdits[name] = le

    # ----------------------- slots -----------------------
    def _onLineEditThresh_textChanged(self, text): self.thresh = float(text)
    def _onLineEditNsave_textChanged (self, text): self.nsave  = float(text)
    def _onLineEditNpass_textChanged (self, text): self.npass  = float(text)

    # ----------------- public functions ----------------
    def open(self):
        self._verifier = MobileFacenet()
        self._verifier.eval()
    
    def isOpened(self):
        self._verifier is not None
    
    def release(self):
        self._verifier = None

    def __call__(self, image):
        """ 
        Params:
            image: {ndarray(H, W, C)}
        Returns:
            feature: {ndarray(256)}
        """
        with torch.no_grad():
            featureL = self._verifier(self._transform(image)).squeeze().numpy()
            featureR = self._verifier(self._transform(image[:, ::-1])).squeeze().numpy()
        feature = np.concatenate([featureL, featureR])
        feature /= np.linalg.norm(feature)

        return feature
    
    def _transform(self, image):
        """
        Params:
            image: {ndarray(H, W, C)}
        Returns:
            tensor: {tensor(1, C, H, W)}
        """
        return torch.from_numpy(np.transpose(image, [2, 0, 1]) - 127.5 / 128.).unsqueeze(0).float()
    
    # -------------------- property --------------------
    @property
    def thresh(self):
        return self._params['Thresh']
    
    @thresh.setter
    def thresh(self, value):
        self._params['Thresh'] = value

    @property
    def nsave(self):
        return int(self._params['N(save)'])
    
    @nsave.setter
    def nsave(self, value):
        self._params['N(save)'] = value

    @property
    def npass(self):
        return int(self._params['N(pass)'])
    
    @npass.setter
    def npass(self, value):
        self._params['N(pass)'] = value

    @property
    def ndim(self):
        return self._ndim


class WidgetPin(_Dialog):

    _names = ['Pin', 'Count']

    def __init__(self, parent=None):
        super(WidgetPin, self).__init__(parent)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._onTimer_timeout)

    def _setup(self):

        self.setWindowTitle("Pin")

        onLineEdits = dict(zip(self._names,[
            self._onLineEditPin_textChanged,
            self._onLineEditCount_textChanged,
        ]))

        x1, y1, w, h, gw, gh = 110, 80, 100, 25, 10, 5

        for i, name in enumerate(self._names):
            lb = QLabel(self);    lb.setGeometry(x1,          y1 + (h + gh) * i, w, h); lb.setText(name)
            le = QLineEdit(self); le.setGeometry(x1 + w + gw, y1 + (h + gh) * i, w, h)
            le.textChanged[str].connect(onLineEdits[name])
            self._qLineEdits[name] = le

    # ----------------------- slots -----------------------
    def _onLineEditPin_textChanged  (self, text): self.pin   = float(text)
    def _onLineEditCount_textChanged(self, text): self.count = float(text)
    def _onTimer_timeout(self): self.release()
    
    # ----------------- public functions ----------------
    def open(self):
        if self.isOpened(): return

        print("Set Pin{:d}".format(self.pin))
        self._timer.start(self.count)
    
    def isOpened(self):
        return self._timer.isActive()
    
    def release(self):
        self._timer.stop()
    
    # -------------------- property --------------------
    @property
    def pin(self):
        return int(self._params['Pin'])
    
    @pin.setter
    def pin(self, value):
        self._params['Pin'] = value
    
    @property
    def count(self):
        return int(self._params['Count'])
    
    @count.setter
    def count(self, value):
        self._params['Count'] = value

