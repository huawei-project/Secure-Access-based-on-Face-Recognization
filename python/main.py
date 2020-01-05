# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-01-04 16:29:49
@LastEditTime : 2020-01-05 20:34:49
@Update: 
'''
import sys
from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow

def main():

    a = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(a.exec_())

if __name__ == '__main__':
    main()