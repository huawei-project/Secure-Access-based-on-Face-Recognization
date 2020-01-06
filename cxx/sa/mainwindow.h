#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QLabel>

#include <darknet.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <malloc.h>
#include <unistd.h>
#include <stdio.h>
#include <dirent.h>
#include <ext/hash_map>
#include <Python.h>

extern "C"{
#include "crop_align.h"
#include "mobilefacenet.h"
#include "mtcnn.h"
#include "util.h"
}
#include "util_cxx.h"
#include "hash_map_string_key.h"

using namespace cv;
using namespace std;
using namespace __gnu_cxx;

typedef hash_map<string, matrix> strmat_map;
typedef hash_map<string, int   > strint_map;
typedef pair<string, matrix>     strmat_pair;
typedef pair<string, float >     strf_pair;
typedef pair<string, int   >     stri_pair;

#define FEATDIR "../../feature"
#define DIM 256

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(int argc, char *argv[], QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void slots_timer_timeout();                                    // 定时采集图片
    void slots_locker_timeout(){unlock(); setPatchStatus(0);}      // 验证锁
    void on_pushButtonSave_clicked();                           // 保存特征

private:
    Ui::MainWindow *ui;
    QTimer *timer, *locker;     // 定时器

    CommandLineParser* parser;  // 命令行解析
    VideoCapture capture;       // 图像采集
    Mat frame;                  // 帧

    network *pnet, *rnet, *onet;// 检测网络
    params detect_params;       // 检测参数
    detect* dets; int nDets;    // 检测结果

    network *mobilefacenet;     // 验证网络
    Mat patch;                  // 人脸图像
    float *feature;             // 人脸特征
    landmark aligned;           // 已对齐特征点

    strmat_map features;        // 所有保存的特征数据

    void setPatchStatus(int status=0);

    // ----------------------- 图像采集相关 -----------------------
    void timerStart(){timerStop(); timer->start(1000 / parser->get<int>("fps"));}
    void timerStop (){if(timer->isActive())timer->stop();}
    void captureOpen   (){captureRelease(); capture.open(parser->get<int>("camera"));}
    void captureRelease(){if(capture.isOpened())capture.release(); }
    void getFrame();
    void showMatInLabel(Mat m, QLabel* label);

    // -------------- 若验证通过，则一段时间内不进行验证 --------------
    int  initLocker(){locker = new QTimer(); connect(locker, SIGNAL(timeout()), this, SLOT(slots_locker_timeout()));return 0;}
    int  deinitLocker(){if(isLocked())unlock(); delete locker;return 0;}
    int  lock(){locker->start(parser->get<int>("lock") * 1000);return 0;}
    int  unlock(){locker->stop();return 0;}
    bool isLocked(){return locker->isActive();}

    // --------------------- 命令行解析及初始化 --------------------
    int parseCommand(int argc, char *argv[]);
    int initCamera();
    int initDetect();
    int initVerify();

    // ------------------------- 释放资源 ------------------------
    int deinitCamera(){timerStop(); captureRelease();delete timer;return 0;}
    int deinitDetect(){free(dets);free_network(pnet);free_network(rnet);free_network(onet);return 0;}
    int deinitVerify(){free(feature);free_network(mobilefacenet);return 0;}

    // ------------------------- 检测验证 -------------------------
    void howAboutThisFrame();                           // 包装代码
    void detectMat(Mat m, detect** dets, int* n);       // 检测人脸
    int  keepCentre(Mat m, detect* dets, int n);        // 保留中心
    Mat  cropAlignMat(Mat m, landmark toAlign, int h=112, int w=96, int mode=1);    // 裁剪对齐
    void generateFeature(Mat m, float* x, int n=256);   // 提取特征

    // ---------------------- 特征比对、保存等 ---------------------
    int     initFeatures();     // 读取特征
    int     deinitFeatures();   // 释放特征
    matrix  findFeatures(string name);  // 按名字获取特征矩阵，若不存在，创建键值对
    string  cmpFeatures();      // 将当前特征`feature`与`features`进行比对
    int     updateFeatures(string name, int index);

    // ------------------------- 引脚相关 ------------------------
    int initPin();
    int triggerPin();
    int deinitPin();
};

#endif // MAINWINDOW_H
