# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 17:09:26
@LastEditTime : 2020-01-04 20:46:26
@Update: 
'''
import os
import cv2
import numpy as np

import torch

from PyQt5.QtWidgets import QDialog, QDesktopWidget
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton

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

    _names = ['Index', 'FPS', 'Width', 'Height']

    def __init__(self, parent=None):
        super(WidgetCamera, self).__init__(parent)

        self._capture = None

    def _setup(self):

        self.setWindowTitle("Camera")

        onLineEdits = dict(zip(self._names,[
            self._onLineEditIndex_textChanged, 
            self._onLineEditFps_textChanged, 
            self._onLineEditWidth_textChanged, 
            self._onLineEditHeight_textChanged, 
        ]))

        x1, y1, w, h, gw, gh = 120, 80, 100, 25, 10, 5

        for i, name in enumerate(self._names):
            lb = QLabel(self);    lb.setGeometry(x1,          y1 + (h + gh) * i, w, h); lb.setText(name)
            le = QLineEdit(self); le.setGeometry(x1 + w + gw, y1 + (h + gh) * i, w, h)
            le.textChanged[str].connect(onLineEdits[name])
            self._qLineEdits[name] = le

    # ----------------------- slots -----------------------
    def _onLineEditIndex_textChanged (self, text): self._params['Index']  = float(text)
    def _onLineEditFps_textChanged   (self, text): self._params['FPS']    = float(text)
    def _onLineEditWidth_textChanged (self, text): self._params['Width']  = float(text)
    def _onLineEditHeight_textChanged(self, text): self._params['Height'] = float(text)

    # ----------------- public functions ----------------
    def open(self):
        """ 打开相机 """
        self._capture = cv2.VideoCapture(int(self._params['Index']))
        self._capture.set(cv2.CAP_PROP_FPS, self._params['FPS'])
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._params['Width'])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._params['Height'])

        return self.isOpened()
    
    def isOpened(self):
        """ 相机是否正常使用 """
        if self._capture is None:
            return False
        return self._capture.isOpened()
    
    def release(self):
        """ 关闭相机 """
        self._capture.release()

    def __call__(self):
        """ 读取图片 """
        return self._capture.read()


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

        x1, y1, w, h, gw, gh = 120, 80, 100, 25, 10, 5

        for i, name in enumerate(self._names):
            lb = QLabel(self);    lb.setGeometry(x1,          y1 + (h + gh) * i, w, h); lb.setText(name)
            le = QLineEdit(self); le.setGeometry(x1 + w + gw, y1 + (h + gh) * i, w, h)
            le.textChanged[str].connect(onLineEdits[name])
            self._qLineEdits[name] = le

    # ----------------------- slots -----------------------
    def _onLineEditMinface_textChanged(self, text): self._params['Minface']  = float(text)
    def _onLineEditP_textChanged(self, text):       self._params['P']        = float(text)
    def _onLineEditR_textChanged(self, text):       self._params['R']        = float(text)
    def _onLineEditO_textChanged(self, text):       self._params['O']        = float(text)

    # ----------------- public functions ----------------
    def open(self):
        self._detector = MtcnnDetector(
            min_face=self._params['Minface'], 
            thresh=[self._params['P'], self._params['R'], self._params['O']]
        )
    
    def isOpened(self):
        self._detector is not None
    
    def release(self):
        self._detector = None

    def __call__(self, image):
        a = self._detector.detect_image(image)
        return self._detector.detect_image(image)


class WidgetVerifier(_Dialog):

    _names = ['Thresh']

    def __init__(self, parent=None):
        super(WidgetVerifier, self).__init__(parent)

        self.open()

    def _setup(self):

        self.setWindowTitle("Verifier")

        onLineEdits = dict(zip(self._names,[
            self._onLineEditThresh_textChanged, 
        ]))

        x1, y1, w, h, gw, gh = 120, 80, 100, 25, 10, 5

        for i, name in enumerate(self._names):
            lb = QLabel(self);    lb.setGeometry(x1,          y1 + (h + gh) * i, w, h); lb.setText(name)
            le = QLineEdit(self); le.setGeometry(x1 + w + gw, y1 + (h + gh) * i, w, h)
            le.textChanged[str].connect(onLineEdits[name])
            self._qLineEdits[name] = le

    # ----------------------- slots -----------------------
    def _onLineEditThresh_textChanged(self, text): self._params['Thresh']  = float(text)

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
            featureL = self._verifier(self._transform(image)).numpy().squeeze()
            featureR = self._verifier(self._transform(image[:, ::-1])).numpy().squeeze()
        feature = np.concatenate([featureL, featureR])
        return feature
    
    def _transform(self, image):
        """
        Params:
            image: {ndarray(H, W, C)}
        Returns:
            tensor: {tensor(1, C, H, W)}
        """
        return torch.from_numpy(np.transpose(image, [2, 0, 1]) - 127.5 / 128.).unsqueeze(0).float()

# if __name__ == '__main__':

#     import sys
#     from PyQt5.QtWidgets import QApplication

#     a = QApplication(sys.argv)
#     w = WidgetDetector()
#     w.show()
#     sys.exit(a.exec_())