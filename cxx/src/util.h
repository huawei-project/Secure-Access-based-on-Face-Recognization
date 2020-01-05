/*
 * @Author: louishsu
 * @Date: 2019-05-31 11:31:24 
 * @Last Modified by:   louishsu
 * @Last Modified time: 2019-05-31 11:31:24 
 */
#ifndef __UTIL_H
#define __UTIL_H

#include <opencv/cv.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <darknet.h>

#define API
#define INPUT
#define OUTPUT
#define INPUTOUTPUT

API IplImage *image_to_ipl(image im);
API image     ipl_to_image(IplImage* src);

image rgb_to_bgr(image im);
image resize_image_scale(image im, float scale);

API static inline float _min(float a, float b){return a<b? a: b;}
API static inline float _max(float a, float b){return a>b? a: b;}
static inline int _ascending(const void * a, const void * b){return ( *(int*)a - *(int*)b );}
static inline int _descending(const void * a, const void * b){return ( *(int*)b - *(int*)a );}

API void find_max_min(INPUT float* x, INPUT int n, OUTPUT float* max, OUTPUT float* min);
API void norm(float* vector, int n);
void normalize_(float* vector, int n);
float dot(float* x1, float* x2, int n);
API float distCosine(float* vec1, float* vec2, int n);
API float distEuclid(float* vec1, float* vec2, int n);

#endif
