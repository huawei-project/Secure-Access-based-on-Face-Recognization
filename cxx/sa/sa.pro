#-------------------------------------------------
#
# Project created by QtCreator 2020-01-06T10:23:21
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = sa
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        crop_align.c \
        main.cpp \
        mainwindow.cpp \
        mobilefacenet.c \
        mtcnn.c \
        util.c \
        util_cxx.cpp

HEADERS += \
        crop_align.h \
        hash_map_string_key.h \
        mainwindow.h \
        mobilefacenet.h \
        mtcnn.h \
        util.h \
        util_cxx.h

########################### OpenCV ###########################
INCLUDEPATH += /usr/local/OpenCV3.4.6/include\
               /usr/local/OpenCV3.4.6/include/opencv\
               /usr/local/OpenCV3.4.6/include/opencv2

LIBS += -L/usr/local/OpenCV3.4.6/lib/ \
        -lopencv_calib3d \
        -lopencv_core \
        -lopencv_dnn \
        -lopencv_features2d \
        -lopencv_flann \
        -lopencv_highgui \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_ml \
        -lopencv_objdetect \
        -lopencv_photo \
        -lopencv_shape \
        -lopencv_stitching \
        -lopencv_superres \
        -lopencv_videoio \
        -lopencv_video \
        -lopencv_videostab

########################### Darknet ###########################
INCLUDEPATH += /usr/local/darknet/include/ \
               /usr/local/darknet/src/

LIBS += -L/usr/local/darknet/lib/ \
        -ldarknet \
        -lm

########################### Python ############################
INCLUDEPATH += /usr/include/python2.7
LIBS += -lpython2.7

FORMS += \
        mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    sa.pro.user
