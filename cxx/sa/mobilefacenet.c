#include <parser.h>
#include <activations.h>
#include "mobilefacenet.h"

API network* load_mobilefacenet()
{
    network* net = load_network("../../cfg/mobilefacenet.cfg",
                    "../../weights/mobilefacenet.weights", 0);
    return net;
}

/*
 * Args:
 *      im: {image} RGB image, range[0, 1]
 * Returns:
 *      cvt:{image} BGR image, range[-1, 1]
 * */
API image convert_mobilefacenet_image(INPUT image im)
{
    int size = im.h*im.w*im.c;

    image cvt = copy_image(im);         // RGB, 0~1
    for (int i = 0; i < size; i++ ){
        float val = im.data[i]*255.;
        val = (val - 127.5) / 128.;
        cvt.data[i] = val; 
    }

    rgbgr_image(cvt);                   // BGR, -1~1
    return cvt;
}

/*
 * 计算人脸特征，输出256维
 * @params:
 * -    im: 输入RGB图像
 * -    mark: 检测到的人脸关键点位置
 * -    X:  特征地址
 */
API void generate_feature(INPUT network* net, INPUT image im, INPUT landmark mark, INPUT landmark aligned, INPUT int mode, OUTPUT float* X)
{
    float* x = NULL;
    image warped = image_aligned_v2(im, mark, aligned, H, W, mode);
    image cvt = convert_mobilefacenet_image(warped);

    x = network_predict(net, cvt.data);
    memcpy(X,     x, N*sizeof(float));

    flip_image(cvt);
    x = network_predict(net, cvt.data);
    memcpy(X + N, x, N*sizeof(float));

    free_image(warped); free_image(cvt);
}

/*
 * Args:
 *      net:    {network*}  MobileFaceNet
 *      im1/2:  {image}     image of size `3 x H x W`
 *      cosine: {float*}    threshold of verification, will be replaced with cosion distance.
 * Returns:
 *      isOne:  {int}       if the same, return 1; else 0.
 * */
API int verify(INPUT network* net, INPUT image im1, INPUT image im2, INPUTOUTPUT float* cosine)
{
    assert(im1.w == W && im1.h == H);
    assert(im2.w == W && im2.h == H);

    float* X;

    float* feat1 = calloc(N*2, sizeof(float));
    image cvt1 = convert_mobilefacenet_image(im1);
    X = network_predict(net, cvt1.data);
    memcpy(feat1, X, N*sizeof(float));
    flip_image(cvt1);
    X = network_predict(net, cvt1.data);
    memcpy(feat1 + N, X, N*sizeof(float));
    
    float* feat2 = calloc(N*2, sizeof(float));
    image cvt2 = convert_mobilefacenet_image(im2);
    X = network_predict(net, cvt2.data);
    memcpy(feat2, X, N*sizeof(float));
    flip_image(cvt2);
    X = network_predict(net, cvt2.data);
    memcpy(feat2 + N, X, N*sizeof(float));

    float dist = distCosine(feat1, feat2, N*2);
    int is_one = dist < *cosine? 0: 1;

    *cosine = dist;

    free(feat1); free(feat2);
    free_image(cvt1); free_image(cvt2);

    return is_one;
}

