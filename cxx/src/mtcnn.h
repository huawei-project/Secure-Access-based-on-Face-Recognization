/*
 * @Author: louishsu
 * @Date: 2019-05-31 11:31:17 
 * @Last Modified by:   louishsu
 * @Last Modified time: 2019-05-31 11:31:17 
 */
#ifndef __MTCNN_H
#define __MTCNN_H

#include <opencv/cv.h>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <math.h>
#include <darknet.h>
#include <list_c.h>
#include <image.h>
#include <activations.h>
#include "include/util.h"


typedef struct bbox
{
    float x1, y1, x2, y2;   /* 左上、右下 */
} bbox;

typedef struct landmark
{
    float x1, y1, x2, y2, x3, y3, x4, y4, x5, y5; /* 左眼、右眼、鼻尖、左嘴角、右嘴角 */
} landmark;

typedef struct detect
{
    float score;    /* 该框评分 */
    bbox bx;        /* 回归方框 */
    bbox offset;    /* 偏置 */
    landmark mk;    /* 位置 */
} detect;

typedef struct params
{
    float min_face;     /* minimal face size */
    float pthresh;      /* threshold of pnet */
    float rthresh;      /* threshold of rnet */
    float othresh;      /* threshold of onet */
    float scale;        /* scale factor */
    int stride;         
    int cellsize;       /* size of cell */
} params;

box bbox_to_box(bbox bbx);
float bbox_area(bbox a);
float bbox_iou(bbox a, bbox b, int mode);

API network* load_mtcnn_net(INPUT char* netname);
API params initParams(INPUT int argc, INPUT char **argv);
API void detect_image(INPUT network *pnet, INPUT network *rnet, INPUT network *onet, INPUT image im, OUTPUT int* n, OUTPUT detect** dets, INPUT params p);
API image draw_detect(INPUT image im, INPUT detect* dets, INPUT int n, INPUT int showscore, INPUT int showbox, INPUT int showmark);
API int keep_one(INPUT detect* dets, INPUT int n, INPUT image im);

#endif
