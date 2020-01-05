#ifndef UTIL_CXX_H
#define UTIL_CXX_H

#include <darknet.h>
#include <opencv2/opencv.hpp>

extern "C"{
#include "include/util.h"
}

using namespace cv;

#define API
#define INPUT
#define OUTPUT
#define INPUTOUTPUT

API Mat image_to_mat(image im);
API image mat_to_image(Mat m);

void saveFloatArray(const String filename, float *X, int n);
void loadFloatArray(const String filename, float *X, int n);

#endif // UTIL_CXX_H
