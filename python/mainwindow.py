# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 16:29:32
@LastEditTime : 2020-01-05 18:19:36
@Update: 
'''
import os
import cv2
import numpy as np
from collections import Counter

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from PyQt5.QtWidgets import QAction, QLabel, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtCore    import QTimer
from PyQt5.QtGui     import QImage, QPixmap

from widget import WidgetCamera, WidgetDetector, WidgetVerifier, WidgetPin
from utils  import imageAlignCrop

class MainWindow(QMainWindow):

    SAVEDIR = '../feature'

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self._setupFunc()
        self._setupUi()

        # ----------- 图像采集 -----------
        self.openCamera()
    
    def _setupFunc(self):
        """ 功能模块初始化 """
        # ------------ 子窗口 ------------
        self._wgCamera   = WidgetCamera  (self)
        self._wgDetector = WidgetDetector(self)
        self._wgVerifier = WidgetVerifier(self)
        self._wgPin      = WidgetPin     (self)

        # ------------ 定时器 ------------
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._onTimer_timeout)

        # ------------- 特征 -------------
        self._feature      = None
        self._savedFatures = self._loadAllFeatures()    # {name[str]: features[ndarray]}

    def _setupUi(self):
        """ UI """
        # ------------ 菜单栏 -----------
        # --------- 设置 ---------
        self._menuSetting = self.menuBar().addMenu('&Setting')
        # ----- 相机 -----
        self._actionCamera = QAction('&Camera', self)
        self._actionCamera.triggered.connect(self._wgCamera.show)
        self._menuSetting.addAction(self._actionCamera)
        # ----- 检测 -----
        self._actionDetector = QAction('&Detector', self)
        self._actionDetector.triggered.connect(self._wgDetector.show)
        self._menuSetting.addAction(self._actionDetector)
        # ----- 验证 -----
        self._actionVerifier = QAction('&Verifier', self)
        self._actionVerifier.triggered.connect(self._wgVerifier.show)
        self._menuSetting.addAction(self._actionVerifier)
        # ----- IO -----
        self._actionPin = QAction('&Pin', self)
        self._actionPin.triggered.connect(self._wgPin.show)
        self._menuSetting.addAction(self._actionPin)

        # ----------- 显示 -----------
        self._lbFrame = QLabel(self)
        self._lbFrame.setGeometry(10, 30, 360, 270)
        self._lbFrame.setPixmap(QPixmap("/home/louishsu/Desktop/jiti.jpg"))

        self._lbPatch = QLabel(self)
        self._lbPatch.setGeometry(380, 30, 96, 112)
        self._lbPatch.setPixmap(QPixmap("/home/louishsu/Desktop/jiti.jpg"))

        # --------- 保存功能 ----------
        self._leSave = QLineEdit(self)
        self._leSave.setGeometry(390, 240, 70, 25)
        self._pbSave = QPushButton('Save', self)
        self._pbSave.setGeometry(390, 270, 70, 25)
        self._pbSave.clicked[bool].connect(self._onPushButtonCamera_clicked)

        # ---------- 主窗口属性 ----------
        self.setGeometry(0, 0, 485, 325)
        self.setWindowTitle("Secure Access")
        s, f = QDesktopWidget().screenGeometry(), self.geometry()
        self.move((s.width() - f.width()) / 2, (s.height() - f.height()) / 2)

        # ------------- 显示 ------------
        self.show()

    # ----------------- custom functions --------------------
    def _timerStart(self):
        """ 启动定时器 """
        self._timer.start(1000 / self._wgCamera.fps)
    
    def _timerStop(self):
        """ 停止定时器 """
        self._timer.stop()

    def _showImageInLabel(self, bgr, label):
        """ 在指定QLabel中显示图片
        Params:
            rgb: {ndarray(H, W, C)} format: BGR
            label: {QLabel}
        """
        if bgr is None:
            bgr = np.zeros((label.height(), label.width(), 3), dtype=np.uint8)
        
        rgb = cv2.cvtColor(
            cv2.resize(bgr, (label.width(), label.height())), cv2.COLOR_BGR2RGB)
        label.setPixmap(
            QPixmap.fromImage(
                QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)))

    def _keepCentralOne(self, frame, boxes):
        """ 返回最中心的人脸索引
        Params:
            frame: {ndarray(H, W, C)}
            boxes: {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
        Returns:
            index: {int}
        """
        n_boxes = boxes.shape[0]
        if n_boxes == 0:
            return -1
            
        centroids = boxes[:, :-1].reshape(n_boxes, 2, 2).mean(axis=1)
        center    = np.array([s / 2 for s in frame.shape[1::-1]])
        distance  = np.linalg.norm(centroids - center, axis=1)
        index     = distance.argmin()
        
        return index

    def _saveFeature(self, feature, name, index=None):
        """ 保存特征
        Params:
            feature: {ndarray(256)}
            name:    {str}
            index:   {int}
        Returns:
            index:   {int}
        """
        name = name.strip()
        if feature is None or len(name) == 0: return None

        savedir = os.path.join(self.SAVEDIR, name)
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        
        if index is None:
            indexes = list(map(lambda x: int(x.split('.')[0]), os.listdir(savedir)))    # 已保存的标签
            rest = [i for i in range(self._wgVerifier.nsave - 1) if i not in indexes]   # 剩余可保存标签
            index = 0 if len(rest) == 0 else rest[0]

        np.savetxt('{:s}/{:d}.txt'.format(savedir, index), feature)
        
        return index

    def _loadFeature(self, name, index):
        """ 载入特征
        Params:
            name:    {str}
            index:   {int}
        Returns:
            feature: {ndarray(256)}
        """
        name = name.strip()
        if len(name) == 0: return None

        savedir = os.path.join(self.SAVEDIR, name)
        if not os.path.exists(savedir): return None

        feature = np.loadtxt('{:s}/{:d}.txt'.format(savedir, index))

        return feature
    
    def _loadAllFeatures(self):
        """ 读取所有特征 """
        allfeatures = dict()

        for name in os.listdir(self.SAVEDIR):

            if name == '.gitignore': continue

            features = np.zeros(shape=(self._wgVerifier.nsave, self._wgVerifier.ndim), dtype=np.float)
            savedir  = os.path.join(self.SAVEDIR, name)

            for featfile in os.listdir(savedir):
                features[int(featfile.split('.')[0])] = np.loadtxt(
                                '{:s}/{:s}'.format(savedir, featfile))
            
            allfeatures[name] = features

        return allfeatures
    
    def _cmpAllFeatures(self, feature):
        """ 当前特征与所有保存特征比对
        Params:
            feature: {ndarray(256)}
        Returns:
            name: {str}
        """
        saved = [(np.array([k] * v.shape[0]), v) for k, v in self._savedFatures.items()]
        if len(saved) == 0: return None, 0
        
        # 获取名字、特征
        names    = np.concatenate([s[0] for s in saved])
        features = np.concatenate([s[1] for s in saved])

        # 计算余弦值
        cosval = np.dot(features, feature)

        # 筛除低相似度
        names  = names [cosval > self._wgVerifier.thresh]
        cosval = cosval[cosval > self._wgVerifier.thresh]

        # 设置验证通过个数
        passed = {k: v for k, v in Counter(names).items() if v > self._wgVerifier.npass}
        if len(passed) == 0: return None, 0

        # 选择通过个数最高的
        n_pass = list(passed.values())
        name = list(passed.keys())[n_pass.index(max(n_pass))]

        return name, max(n_pass)

    def openCamera(self):
        """ 打开相机 """
        if self._wgCamera.open():
            self._timerStart()
        else:
            QMessageBox.warning(self, '', 
                '摄像头{:d}打开错误！'.format(self._wgCamera.index), 
                QMessageBox.Ok)
    
    def readImageFromCamera(self):
        """ 读取相机 """
        status, frame = self._wgCamera()
        
        if not status:
            self._timer.stop(); self._wgCamera.release()
            QMessageBox.warning(self, '', '无法读取图像！', QMessageBox.Ok)
            return

        shape = [int(s * self._wgCamera.scale) for s in frame.shape[1::-1]]
        frame = cv2.resize(frame, tuple(shape))

        return frame
    
    def detectAndExtract(self, image):
        """ 检测、提取特征
        Params:
            image: {ndarray(H, W, C)}
        Returns:
            feature: {ndarray(256)}
        """
        boxes, landmark = self._wgDetector(image)   # 检测
        index = self._keepCentralOne(image, boxes)  # 保留最中心
        if index == -1: return None, None
        patch = imageAlignCrop(image, 
                    landmark[index].reshape(-1, 2)) # 截取
        feature = self._wgVerifier(patch)           # 提取特征

        return patch, feature

    # ------------------------ slots ------------------------
    def _onTimer_timeout(self):
        """ 定时器触发 """
        self._timerStop()
        
        # 读取图像
        frame = self.readImageFromCamera()
        self._showImageInLabel(frame, self._lbFrame)
        
        # 检测、提取特征
        patch, feature = self.detectAndExtract(frame)
        self._showImageInLabel(patch, self._lbPatch)
        
        if feature is not None:
            # 保存最新的特征
            self._feature = feature.copy()

            # 与所有已保存特征进行比对
            name, n = self._cmpAllFeatures(feature)

            if name is not None:    # 验证通过
                
                # 更新特征
                index = self._wgVerifier.nsave - 1
                self._saveFeature(feature, name, index=index)
                self._savedFatures[name][index] = feature

                # 开门
                self._wgPin.open()

                self.statusBar().showMessage('你好！{:s}！({:d})'.format(name, n))
            else:                   # 验证未通过
                self.statusBar().showMessage('验证未通过！({:d})'.format(n))
        else:
            self.statusBar().clearMessage()
        
        self._timerStart()

    def _onPushButtonCamera_clicked(self):
        """ 保存当前特征 """
        self._timerStop()

        name = self._leSave.text()
        
        # 保存并更新特征
        index = self._saveFeature(self._feature, name)
        self._savedFatures = self._loadAllFeatures()

        # 显示状态栏
        if index is not None:
            self.statusBar().showMessage('特征({:s}/{:d})已保存！'.format(name, index))
        else:
            self.statusBar().showMessage('特征保存失败，请输入用户名')
        
        self._timerStart()
