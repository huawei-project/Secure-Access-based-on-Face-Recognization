/*
 * @Author: louishsu
 * @Date: 2019-05-31 11:31:08 
 * @Last Modified by:   louishsu
 * @Last Modified time: 2019-05-31 11:31:08 
 */
#ifndef __MOBILEFACENET_H
#define __MOBILEFACENET_H

#include <opencv/cv.h>
#include <darknet.h>
#include "include/mtcnn.h"
#include "include/util.h"
#include "include/crop_align.h"

#define H 112
#define W 96
#define N 128

API network* load_mobilefacenet();
API image convert_mobilefacenet_image(INPUT image im);
API void generate_feature(INPUT network* net, INPUT image im, INPUT landmark mark, INPUT landmark aligned, INPUT int mode, OUTPUT float* X);
API int verify(INPUT network* net, INPUT image im1, INPUT image im2, INPUTOUTPUT float* cosine);

#endif
