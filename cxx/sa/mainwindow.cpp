#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(int argc, char *argv[], QWidget *parent):
    QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    if(parseCommand(argc, argv))
        return;

    if(initCamera())
        ui->statusBar->showMessage("Can't open camera!", 3000);
    initLocker();
    initDetect();
    initVerify();
    initFeatures();
    initPin();
}

MainWindow::~MainWindow()
{
    deinitCamera();
    deinitLocker();
    deinitVerify();
    deinitDetect();
    deinitFeatures();
    deinitPin();
    delete ui;
}

void MainWindow::on_timer_timeout()
{
    timerStop();

    getFrame();
    howAboutThisFrame();

    showMatInLabel(frame, ui->labelFrame);
    showMatInLabel(patch, ui->labelPatch);

    timerStart();
}

void MainWindow::on_pushButtonSave_clicked()
{
    string name = ui->lineEditName->text().toStdString();
    if (name.empty()){
        ui->statusBar->showMessage("请输入用户名！");
        return;
    }

    int index = 1;
    for (; index < parser->get<int>("nsave"); index++){
        char featfile[256];
        sprintf(featfile, "%s/%s %d", FEATDIR, name.c_str(), index);
        if(access(featfile, F_OK)) break;
    }
    if (index == parser->get<int>("nsave")) index = 1;

    updateFeatures(name, index);
    ui->statusBar->showMessage("特征已保存!");
}

void MainWindow::setPatchStatus(int status)
{
    string style;
    switch (status) {
    case 1:
        style = "border: 5px solid blue;";  break;
    case 2:
        style = "border: 5px solid red;" ;  break;
    default:
        style = "border: 1px solid black;"; break;
    }
    this->ui->labelPatch->setStyleSheet(QString::fromStdString(style));
}

void MainWindow::showMatInLabel(Mat m, QLabel* label)
{
    if (m.empty())
        m = Mat::zeros(label->height(), label->width(), CV_8UC3);

    Mat frame = m.clone();
    cvtColor(frame, frame, CV_BGR2RGB);
    QImage image = QImage((const unsigned char*)(frame.data),
                          frame.cols, frame.rows, frame.cols*frame.channels(),
                          QImage::Format_RGB888).scaled(label->width(), label->height());
    label->setPixmap(QPixmap::fromImage(image));
    frame.release();
}

void MainWindow::getFrame()
{
    capture >> frame;
    cv::resize(frame, frame, Size(),
               parser->get<double>("fx"), parser->get<double>("fx"));
}

int MainWindow::parseCommand(int argc, char *argv[])
{
    parser = new CommandLineParser(argc, argv,
        "{ help h usage ?   |               | 显示帮助      }"
        "{ camera c         | 0             | 相机索引      }"
        "{ fps              | 30            | 帧率          }"
        "{ fx               | 0.5           | 图像缩放尺度  }"
        "{ minface m        | 96            | 最小检测人脸尺寸  }"
        "{ threshp p        | 0.8           | PNet 阈值 }"
        "{ threshr r        | 0.8           | RNet 阈值 }"
        "{ thresho o        | 0.8           | ONet 阈值 }"
        "{ thresh t         | 0.75          | 验证阈值  }"
        "{ nsave            | 5             | 保存的特征总个数  }"
        "{ npass            | 3             | 通过特征个数      }"
        "{ lock l           | 5             | 锁时间        }"
        "{ pin              | 13            | 引脚        }");
    if(parser->has("help")){
        parser->printMessage(); return 1;
    } else return 0;
}

int MainWindow::initCamera()
{
    timer = new QTimer(); connect(timer, SIGNAL(timeout()), this, SLOT(on_timer_timeout()));

    frame = Mat::zeros(ui->labelFrame->height(), ui->labelFrame->width(), CV_8UC3);
    patch = Mat::zeros(ui->labelPatch->height(), ui->labelPatch->width(), CV_8UC3);
    showMatInLabel(frame, ui->labelFrame); showMatInLabel(patch, ui->labelPatch);

    captureOpen(); timerStart();

    if (!capture.isOpened()) return 1;
    return 0;
}

int MainWindow::initDetect()
{
    pnet = load_mtcnn_net((char*)"PNet");
    rnet = load_mtcnn_net((char*)"RNet");
    onet = load_mtcnn_net((char*)"ONet");
    dets  = (detect*)calloc(0, sizeof(detect)); nDets = 0;

    detect_params.min_face  = parser->get<float>("minface");
    detect_params.scale     = 0.79;
    detect_params.pthresh   = parser->get<float>("threshp");
    detect_params.rthresh   = parser->get<float>("threshr");
    detect_params.othresh   = parser->get<float>("thresho");
    detect_params.cellsize  = 12;
    detect_params.stride    = 2;

    return 0;
}

int MainWindow::initVerify()
{
    mobilefacenet = load_mobilefacenet();
    feature = (float*)calloc(DIM, sizeof(float));

    aligned.x1 = 3.0290000915527344e+01; aligned.y1 = 5.1700000762939453e+01;
    aligned.x2 = 6.5529998779296875e+01; aligned.y2 = 5.1500000000000000e+01;
    aligned.x3 = 4.8029998779296875e+01; aligned.y3 = 7.1739997863769531e+01;
    aligned.x4 = 3.3549999237060547e+01; aligned.y4 = 9.2370002746582031e+01;
    aligned.x5 = 6.2729999542236328e+01; aligned.y5 = 9.2199996948242188e+01;

    return 0;
}

void MainWindow::howAboutThisFrame()
{
    Mat m = frame.clone();

    // 检测
    detectMat(m, &dets, &nDets);
    if (nDets == 0){
        patch = Mat(); memset(feature, 0, DIM);
        return;
    }

    // 若已锁，则不进行验证
    if(isLocked()) return;

    // 验证
    int index = keepCentre(m, dets, nDets);
    patch = cropAlignMat(m, dets[index].mk);
    generateFeature(patch, feature);
    m.release();

    // 特征比对，若无匹配，则返回`""`
    string name = cmpFeatures();
    if(name == ""){
        setPatchStatus(2);
        return;
    }

    // 验证通过，锁住，并更新特征
    lock(); setPatchStatus(1);
    updateFeatures(name, 0);
    triggerPin();
    ui->statusBar->showMessage(QString::fromStdString("Hello! " + name), 3000);
}

void MainWindow::detectMat(Mat m, detect** dets, int* n)
{
    image im = mat_to_image(m);
    detect_image(pnet, rnet, onet, im, n, dets, detect_params);
    free_image(im);
}

int MainWindow::keepCentre(Mat m, detect* dets, int n)
{
    image im = mat_to_image(m);
    int index = keep_one(dets, n, im);
    free_image(im);
    return index;
}

Mat MainWindow::cropAlignMat(Mat m, landmark toAlign, int h, int w, int mode)
{
    image im = mat_to_image(m);

    //! 对齐
    image patch = image_aligned_v2(im, toAlign, aligned, h, w, mode);
    Mat matCrop = image_to_mat(patch);

    //! 释放中间变量
    free_image(im); free_image(patch);

    return matCrop;
}

void MainWindow::generateFeature(Mat m, float* x, int n)
{
    n /= 2;

    //! 将当前图片转换为符合`mobilefacenet`输入的图片，像素值范围(-1, 1)
    image im = mat_to_image(m);
    image cvt = convert_mobilefacenet_image(im);

    //! 前向计算，并赋值前128个单位的内存空间
    float* pf = network_predict(mobilefacenet, cvt.data);
    memcpy(x,     pf, n*sizeof(float));

    //! 左右翻转
    flip_image(cvt);

    //! 前向计算，并赋值后128个单位的内存空间
    pf = network_predict(mobilefacenet, cvt.data);
    memcpy(x + n, pf, n*sizeof(float));

    //! 单位化
    normalize_(x, n*2);

    free_image(im); free_image(cvt);
}

int MainWindow::initFeatures()
{
    DIR *dp;
    struct dirent *dirp;

    dp = opendir(FEATDIR);
    while((dirp = readdir(dp)) != nullptr)
    {
        if(dirp->d_name[0] == '.') continue;

        char name[32]; int index=0;
        sscanf(dirp->d_name, "%s %d", name, &index);

        matrix m = findFeatures(name);
        if (index < parser->get<int>("nsave")){
            char featfile[256];
            sprintf(featfile, "%s/%s %d", FEATDIR, name, index);
            loadFloatArray(featfile, m.vals[index], DIM);
        }
    }
    closedir(dp);
}

int MainWindow::deinitFeatures()
{
    strmat_map::iterator it;
    for(it = features.begin(); it != features.end(); it++)
        free_matrix(it->second);
    return 0;
}

matrix MainWindow::findFeatures(string name)
{
    if (features.find(name) == features.end())
        features.insert(
            strmat_pair(name, make_matrix(parser->get<int>("nsave"), DIM)));
    return features[name];
}

string MainWindow::cmpFeatures()
{
    strint_map sim;
    for(strmat_map::iterator it = features.begin(); it != features.end(); it++){
        string name = it->first; matrix m = it->second;     // 获取名字、及其保存的特征

        sim.insert(stri_pair(name, 0));                     // 用于统计通过的个数
        for(int i = 0; i < parser->get<int>("nsave"); i++){
            float cosval = dot(feature, m.vals[i], DIM);    // 计算余弦值
            if(cosval > parser->get<float>("thresh"))       // 大于阈值，计数自增
                sim[name] += 1;
        }
    }

    // 统计通过数目最高的，作为匹配
    string name = ""; int npass = -1;
    for(strint_map::iterator it = sim.begin(); it != sim.end(); it++){
        int num = it->second;
        if ((num > parser->get<int>("npass")) && (num > npass)){    // 滤除通过个数少于`npass`的
            name = it->first; npass = num;
        }
    }
    return name;
}

int MainWindow::updateFeatures(string name, int index)
{
    matrix m = findFeatures(name);
    if(index > parser->get<int>("nsave") - 1) return 1;

    // 更新RAM
    memcpy(m.vals[index], feature, DIM);

    // 保存到ROM
    char featfile[256];
    sprintf(featfile, "%s/%s %d", FEATDIR, name.c_str(), index);
    saveFloatArray(featfile, feature, DIM);
    ui->statusBar->showMessage("特征已保存!");
}


int MainWindow::initPin()
{
    if(parser->get<int>("pin") < 0) return 1;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/opt/nvidia/jetson-gpio/lib/python/')");
    PyRun_SimpleString("sys.path.append('/opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO')");
    PyRun_SimpleString("import Jetson.GPIO as GPIO");
    PyRun_SimpleString("GPIO.setmode(GPIO.BOARD)");
    string s = "pin=" + parser->get<string>("pin");
    PyRun_SimpleString(s);
    PyRun_SimpleString("GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)");

    return 0;
}

int MainWindow::triggerPin()
{
    if(parser->get<int>("pin") < 0) return 1;

    PyRun_SimpleString("GPIO.output(pin, GPIO.LOW)");
    sleep(1);
    PyRun_SimpleString("GPIO.output(pin, GPIO.HIGH)");

    return 0;
}

int MainWindow::deinitPin()
{
    if(parser->get<int>("pin") < 0) return 1;

    PyRun_SimpleString("GPIO.cleanup");
    Py_Finalize();

    return 0;
}
