# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 17:09:26
@LastEditTime : 2020-01-04 20:05:27
@Update: 
'''
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import QDialog, QDesktopWidget
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton


class _Dialog(QDialog):

    def __init__(self, parent=None):
        super(_Dialog, self).__init__(parent)
    
        self._params = dict()
        self._qLineEdits = dict()
        
        self._file  = '../param/{}.txt'.format(self.__class__.__name__)
        if os.path.exists(self._file): self._load()
        
        self._baseUi()
        self._setupUi()
    
    def _baseUi(self):

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

    def _setupUi(self):
        
        self._setup()
        self._show()

    def _setup(self):

        raise NotImplementedError

    def _show(self):
        for k, v in self._params.items():
            self._qLineEdits[k].setText(str(v))

    def _save(self):

        s = '\n'.join([' '.join([k, str(v)]) for k, v in self._params.items()])
        with open(self._file, 'w') as f:
            f.write(s)

    def _load(self):

        with open(self._file, 'r') as f:
            s = f.read()
        self._params = {k: float(v) for k, v in [l.split(' ') for l in s.split('\n')]}

    def _onPushButtonSave_clicked(self):

        self._save()

    def _onPushButtonApply_clicked(self):

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

    def _onPushButtonApply_clicked(self):
        """ 应用参数，重启相机，并关闭窗口 """
        if self.isOpened():
            self.release()
            self.open()
        self.close()

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

    def read(self):
        """ 读取图片 """
        return self._capture.read()


if __name__ == '__main__':

    import sys
    from PyQt5.QtWidgets import QApplication

    a = QApplication(sys.argv)
    w = WidgetCamera()
    w.show()
    sys.exit(a.exec_())