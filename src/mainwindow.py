# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 16:29:32
@LastEditTime : 2020-01-04 21:03:52
@Update: 
'''
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore    import QTimer

from widget import WidgetCamera, WidgetDetector, WidgetVerifier

class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self._widgetCamera   = WidgetCamera(self)
        self._widgetDetector = WidgetDetector(self)
        self._widgetVerifier = WidgetVerifier(self)
        self._setupUi()

    def _setupUi(self):
        """ UI """

        # ------------ 菜单栏 -----------
        self.menubar = self.menuBar()

        # --------- 设置 ---------
        self.menuSetting = self.menubar.addMenu('&Setting')
        # ----- 相机 -----
        self.actionCamera = QAction('&Camera', self)
        self.actionCamera.triggered.connect(self._widgetCamera.show)
        self.menuSetting.addAction(self.actionCamera)
        # ----- 检测 -----
        self.actionDetector = QAction('&Detector', self)
        self.actionDetector.triggered.connect(self._widgetDetector.show)
        self.menuSetting.addAction(self.actionDetector)
        # ----- 验证 -----
        self.actionVerifier = QAction('&Verifier', self)
        self.actionVerifier.triggered.connect(self._widgetVerifier.show)
        self.menuSetting.addAction(self.actionVerifier)

        # ---------- 主窗口属性 ----------
        self.setGeometry(0, 0, 480, 360)
        self.setWindowTitle("Secure Access")
        s, f = QDesktopWidget().screenGeometry(), self.geometry()
        self.move((s.width() - f.width()) / 2, (s.height() - f.height()) / 2)

        # ------------- 显示 ------------
        self.show()


if __name__ == '__main__':

    import sys
    from PyQt5.QtWidgets import QApplication

    a = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(a.exec_())