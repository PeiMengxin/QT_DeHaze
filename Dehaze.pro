#-------------------------------------------------
#
# Project created by QtCreator 2016-08-30T16:11:11
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Dehaze
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    dehazethread.cpp \
    dehazeclass.cpp

HEADERS  += mainwindow.h \
    dehazethread.h \
    dehazeclass.h

FORMS    += mainwindow.ui

INCLUDEPATH += E:/opencv/build/include

#LIBS    += -LE:/opencv/build/x64/vc12/lib \
#            opencv_calib3d249d.lib \
#            opencv_contrib249d.lib \
#            opencv_core249d.lib \
#            opencv_features2d249d.lib \
#            opencv_flann249d.lib \
#            opencv_gpu249d.lib \
#            opencv_highgui249d.lib \
#            opencv_imgproc249d.lib \
#            opencv_legacy249d.lib \
#            opencv_ml249d.lib \
#            opencv_nonfree249d.lib \
#            opencv_objdetect249d.lib \
#            opencv_ocl249d.lib \
#            opencv_photo249d.lib \
#            opencv_stitching249d.lib \
#            opencv_superres249d.lib \
#            opencv_ts249d.lib \
#            opencv_video249d.lib \
#            opencv_videostab249d.lib

LIBS    += -LE:/opencv/build/x64/vc12/lib \
            opencv_calib3d249.lib \
            opencv_contrib249.lib \
            opencv_core249.lib \
            opencv_features2d249.lib \
            opencv_flann249.lib \
            opencv_gpu249.lib \
            opencv_highgui249.lib \
            opencv_imgproc249.lib \
            opencv_legacy249.lib \
            opencv_ml249.lib \
            opencv_nonfree249.lib \
            opencv_objdetect249.lib \
            opencv_ocl249.lib \
            opencv_photo249.lib \
            opencv_stitching249.lib \
            opencv_superres249.lib \
            opencv_ts249.lib \
            opencv_video249.lib \
            opencv_videostab249.lib