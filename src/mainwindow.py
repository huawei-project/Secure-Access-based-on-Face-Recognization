# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 16:29:32
@LastEditTime : 2020-01-04 17:28:59
@Update: 
'''
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from PyQt5.QtWidgets import QAction

class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self._setupUi()
    
    def _setupUi(self):
        """ UI """

        # ------------ 菜单栏 -----------
        # menubar = self.menuBar()
        # # ---- 相机 ----
        # menuCamera = menubar.addMenu('&Camera')
        # actionCamera = QAction(self)

        # ---------- 主窗口属性 ----------
        self.setGeometry(0, 0, 200, 200)
        self.setWindowTitle("Secure Access")
        self._moveToCenter()

        # ------------- 显示 ------------
        self.show()

    def _moveToCenter(self):
        """ 居中 """
        
        s, f = QDesktopWidget().screenGeometry(), self.geometry()
        self.move((s.width() - f.width()) / 2, (s.height() - f.height()) / 2)

if __name__ == '__main__':

    import sys
    from PyQt5.QtWidgets import QApplication

    a = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(a.exec_())