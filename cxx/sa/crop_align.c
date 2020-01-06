#include "crop_align.h"

void printCvMat(CvMat* M)
{
    for (int r = 0; r < M->rows; r++){
        for (int c = 0; c < M->cols; c++){
            printf("%.2f ", M->data.fl[r*M->cols + c]);
        }
        printf("\n");
    }
        printf("\n\n");
}

/*
 * @param
 *      xy: [x, y]， Nx2
 * @return
 *      ret: 2Nx4
 * @notes
 *      x  y 1 0
 *      y -x 0 1
 */
CvMat* _stitch(const CvMat* xy)
{
    int rows = xy->rows;
    CvMat* C = cvCreateMat(rows, 1, CV_32F);

    // x, y, 1, 0
    cvGetCol(xy, C, 0); CvMat* x = cvCloneMat(C);
    cvGetCol(xy, C, 1); CvMat* y = cvCloneMat(C);
    CvMat* ones  = cvCreateMat(rows, 1, CV_32F);
    CvMat* zeros = cvCreateMat(rows, 1, CV_32F);
    for (int i = 0; i < rows; i++){
        ones ->data.fl[i] = 1.; zeros->data.fl[i] = 0.;
    }

    CvMat* X = cvCreateMat(2*xy->rows, 4, CV_32F);
    cvGetSubRect(X, C, cvRect(0,    0, 1, rows)); cvCopy(    x, C, NULL);
    cvGetSubRect(X, C, cvRect(1,    0, 1, rows)); cvCopy(    y, C, NULL);
    cvGetSubRect(X, C, cvRect(2,    0, 1, rows)); cvCopy( ones, C, NULL);
    cvGetSubRect(X, C, cvRect(3,    0, 1, rows)); cvCopy(zeros, C, NULL);
    for (int i = 0; i < rows; i++) x->data.fl[i] *= -1.;
    cvGetSubRect(X, C, cvRect(0, rows, 1, rows)); cvCopy(    y, C, NULL);
    cvGetSubRect(X, C, cvRect(1, rows, 1, rows)); cvCopy(    x, C, NULL);
    cvGetSubRect(X, C, cvRect(2, rows, 1, rows)); cvCopy(zeros, C, NULL);
    cvGetSubRect(X, C, cvRect(3, rows, 1, rows)); cvCopy( ones, C, NULL);

    cvReleaseMat(&C);
    cvReleaseMat(&x); cvReleaseMat(&y);
    cvReleaseMat(&ones); cvReleaseMat(&zeros);

    return X;
}


/*
 * @param
 *      M:  2x3
 *      uv: [u, v]， Nx2
 * @return
 *      xy: Nx2
 * @notes
 *      xy = [uv, 1] * M^T, Nx2
 */
CvMat* _tformfwd(const CvMat* M, const CvMat* uv)
{
    int rows = uv->rows;
    int cols = uv->cols;
    CvMat* mat = cvCreateMat(rows, cols, CV_32F);

    CvMat* UV = cvCreateMat(rows, cols + 1, CV_32F);

    cvGetSubRect(UV, mat, cvRect(0, 0, cols, rows));

    cvCopy(uv, mat, NULL);
    for (int r = 0; r < rows; r++){
        UV->data.fl[r*(cols+1) + cols] = 1.;
    }

    CvMat* MT = cvCreateMat(M->cols, M->rows, CV_32F);
    CvMat* xy = cvCreateMat(rows, cols, CV_32F);
    cvTranspose(M, MT);
    cvMatMul(UV, MT, xy);

    cvReleaseMat(&UV); cvReleaseMat(&mat); cvReleaseMat(&MT);
    return xy;
}

float _matrixNorm2(CvMat* Mat)
{
    CvMat* U = cvCreateMat(Mat->rows, Mat->rows, CV_32F);
    CvMat* W = cvCreateMat(Mat->rows, Mat->cols, CV_32F);
    CvMat* V = cvCreateMat(Mat->cols, Mat->cols, CV_32F);

    cvSVD(Mat, W, U, V, CV_SVD_V_T);

    float s = FLT_MIN;
    for (int i = 0; i < W->rows*W->cols; i++){
        float val = W->data.fl[i];
        if (val > s){
            s = val;
        }
    }

    cvReleaseMat(&U); cvReleaseMat(&W); cvReleaseMat(&V);
    return s;
}

/*
 * @param
 *      uv: [u, v]， Nx2
 *      xy: [x, y]， Nx2
 * @return
 * @notes
 * -    Xr = Y   ===>  r = (X^T X + \lambda I)^{-1} X^T Y
 */
CvMat* _findNonreflectiveSimilarity(const CvMat* uv, const CvMat* xy)
{
    CvMat* X = _stitch(xy);                         // 2N x  4

    CvMat* XT = cvCreateMat(X->cols, X->rows, CV_32F);
    cvTranspose(X, XT);                             //  4 x 2N

    CvMat* XTX = cvCreateMat(XT->rows, X->cols, CV_32F);
    cvMatMul(XT, X, XTX);                           //  4 x  4
    for (int i = 0; i < XTX->rows; i++) XTX->data.fl[i*XTX->rows + i] += 1e-15;

    CvMat* XTXi = cvCreateMat(XTX->rows, XTX->cols, CV_32F);
    cvInvert(XTX, XTXi, CV_LU);                     //  4 x  4

    // -----------------------------------------------------------------------

    CvMat* uvT = cvCreateMat(uv->cols, uv->rows, CV_32F);
    cvTranspose(uv, uvT);                           //  2 x  N
    CvMat header;
    CvMat* YT = cvReshape(uvT, &header, 0, 1);      //  1 x 2N    TODO
    CvMat* Y = cvCreateMat(YT->cols, YT->rows, CV_32F);
    cvTranspose(YT, Y);                             // 2N x  1

    CvMat* XTXiXT = cvCreateMat(XTXi->rows, XT->cols, CV_32F);
    CvMat* r = cvCreateMat(XTXiXT->rows, Y->cols, CV_32F);
    cvMatMul(XTXi, XT, XTXiXT); cvMatMul(XTXiXT, Y, r);       //  4 x  1

    // -----------------------------------------------------------------------

    cvReleaseMat(&X); cvReleaseMat(&XT);
    cvReleaseMat(&XTX); cvReleaseMat(&XTXi); cvReleaseMat(&XTXiXT);
    cvReleaseMat(&uvT); cvReleaseMat(&Y);

    // =======================================================================

    CvMat* R = cvCreateMat(3, 3, CV_32F);
    R->data.fl[0 * 3 + 0] = r->data.fl[0]; R->data.fl[0 * 3 + 1] = -r->data.fl[1]; R->data.fl[0 * 3 + 2] = 0.;
    R->data.fl[1 * 3 + 0] = r->data.fl[1]; R->data.fl[1 * 3 + 1] =  r->data.fl[0]; R->data.fl[1 * 3 + 2] = 0.;
    R->data.fl[2 * 3 + 0] = r->data.fl[2]; R->data.fl[2 * 3 + 1] =  r->data.fl[3]; R->data.fl[2 * 3 + 2] = 1.;

    CvMat* Ri = cvCreateMat(R->cols, R->rows, CV_32F);
    cvInvert(R, Ri, CV_LU);

    CvMat* MT = cvCreateMat(3, 2, CV_32F);
    cvGetSubRect(Ri, MT, cvRect(0, 0, 2, 3));

    CvMat* M = cvCreateMat(MT->cols, MT->rows, CV_32F);
    cvTranspose(MT, M);

    // -----------------------------------------------------------------------

    cvReleaseMat(&r); cvReleaseMat(&R); cvReleaseMat(&Ri); cvReleaseMat(&MT);

    return M;
}

/*
 * @param
 *      uv: [u, v]， Nx2
 *      xy: [x, y]， Nx2
 * @return
 * @notes
 */
CvMat* _findReflectiveSimilarity(const CvMat* uv, const CvMat* xy)
{
    CvMat* xyR = cvCloneMat(xy);
    for (int r = 0; r < xyR->rows; r++) xyR->data.fl[r*xyR->cols] *= -1;

    CvMat* M1 = _findNonreflectiveSimilarity(uv, xy);
    CvMat* M2 = _findNonreflectiveSimilarity(uv, xyR);

    cvReleaseMat(&xyR);

    for (int r = 0; r < M2->rows; r++) M2->data.fl[r*M2->cols] *= -1;

    CvMat* xy1 = _tformfwd(M1, uv);
    CvMat* xy2 = _tformfwd(M2, uv);
    cvSub(xy1, xy, xy1, NULL); cvSub(xy2, xy, xy2, NULL);

    float norm1 = _matrixNorm2(xy1);
    float norm2 = _matrixNorm2(xy2);

    cvReleaseMat(&xy1); cvReleaseMat(&xy2);

    if (norm1 < norm2){
        cvReleaseMat(&M2);
        return M1;
    } else {
        cvReleaseMat(&M1);
        return M2;
    }
}

/*
 * @param
 *      src: 原始坐标点 [[x1, y1], [x2, y2], ..., [xn, yn]]
 *      dst: 对齐对标点 [[x1, y1], [x2, y2], ..., [xn, yn]]
 *      mode:模式
 * @return
 * @notes
 */
CvMat* cp2form(const CvMat* src, const CvMat* dst, int mode)
{
    CvMat* M;

    if (mode == 0){
        M = _findNonreflectiveSimilarity(src, dst);
    } else if (mode == 1){
        M = _findReflectiveSimilarity(src, dst);
    } else {
        printf("Mode %d not supported!\n", mode);
    }

    return M;
}

// 1.02 -0.09 0.88
// 0.09 1.02 7.36


// 0.92 -0.01 -2.13
// 0.01 0.92 17.92


/*
 * @returns {landmark}
 * @notes 
 * -    image size: 112 x 96
 */
landmark initAligned()
{
    landmark aligned = {0};

    aligned.x1 = 30.2946; aligned.y1 = 51.6963;
    aligned.x2 = 65.5318; aligned.y2 = 51.5014;
    aligned.x3 = 48.0252; aligned.y3 = 71.7366;
    aligned.x4 = 33.5493; aligned.y4 = 92.3655;
    aligned.x5 = 62.7299; aligned.y5 = 92.2041;

    return aligned;
}

/*
 * @returns {landmark}
 * @notes 
 * -    image size: 112 x 96
 */
landmark initAlignedOffset()
{
    int h = 112, w = 96;
    landmark offset = {0};

    offset.x1 = 30.2946 / w; offset.y1 = 51.6963 / h;
    offset.x2 = 65.5318 / w; offset.y2 = 51.5014 / h;
    offset.x3 = 48.0252 / w; offset.y3 = 71.7366 / h;
    offset.x4 = 33.5493 / w; offset.y4 = 92.3655 / h;
    offset.x5 = 62.7299 / w; offset.y5 = 92.2041 / h;

    return offset;
}

/*
 * @params
 * -    offset: 相对于边长h, w的偏置
 * -    h, w:   边长
 * @returns {landmark}
 */
landmark dstLandmark(landmark offset, float h, float w)
{
    landmark dst = {0};
    
    dst.x1 = offset.x1 * w; dst.y1 = offset.y1 * h;
    dst.x2 = offset.x2 * w; dst.y2 = offset.y2 * h;
    dst.x3 = offset.x3 * w; dst.y3 = offset.y3 * h;
    dst.x4 = offset.x4 * w; dst.y4 = offset.y4 * h;
    dst.x5 = offset.x5 * w; dst.y5 = offset.y5 * h;

    return dst;
}

/*
 * @params
 * -    mark: 原图中的坐标
 * -    x, y: 该关键点所在回归框的左上角坐标
 * @return
 *      mark: 在回归框中的坐标
 */
landmark substract_offset(landmark mark, float x, float y)
{
    mark.x1 -= x; mark.y1 -= y;
    mark.x2 -= x; mark.y2 -= y;
    mark.x3 -= x; mark.y3 -= y;
    mark.x4 -= x; mark.y4 -= y;
    mark.x5 -= x; mark.y5 -= y;
    return mark;
}

/*
 * @params
 * -    mark: 原图中的坐标
 * -    scale: 缩放比例
 * @return
 *      mark: 在回归框(h, w)中的坐标
 */
landmark multiply_scale(landmark mark, float scale)
{
    mark.x1 *= scale; mark.y1 *= scale;
    mark.x2 *= scale; mark.y2 *= scale;
    mark.x3 *= scale; mark.y3 *= scale;
    mark.x4 *= scale; mark.y4 *= scale;
    mark.x5 *= scale; mark.y5 *= scale;
    return mark;
}

CvMat* landmark_to_cvMat(landmark mark)
{
    CvMat* mat = cvCreateMat(5, 2, CV_32FC1);
    mat->data.fl[0] = mark.x1; mat->data.fl[1] = mark.y1;
    mat->data.fl[2] = mark.x2; mat->data.fl[3] = mark.y2;
    mat->data.fl[4] = mark.x3; mat->data.fl[5] = mark.y3;
    mat->data.fl[6] = mark.x4; mat->data.fl[7] = mark.y4;
    mat->data.fl[8] = mark.x5; mat->data.fl[9] = mark.y5;
    return mat;
}

/*
 * @params  
 * -    im:     原图
 * -    box:    回归框
 * -    h, w:   期望得到的图片尺寸
 * -    scale:  原图(hh, ww)缩放到(h, w)的比例
 * @returns
 *      resized: 切割后的图像
 * @notes
 * -    若h == 0, w == 0则不进行resize
 */
image image_crop(image im, bbox box, int h, int w, float* scale)
{
    float cx = (box.x2 + box.x1) / 2;   // centroid
    float cy = (box.y2 + box.y1) / 2;

    // padding
    float w_ = box.x2 - box.x1 + 1;
    float h_ = box.y2 - box.y1 + 1;

    h = (h == 0)? h_: h; w = (w == 0)? w_: w;

    float ratio_src = h_ / w_;
    float ratio_dst = (float)h / (float)w;
    int ww_ = 0, hh_ = 0;
    if (ratio_src < ratio_dst){
        // 原图h为较短边，以h为基准截取 h/w 比例的人脸
        hh_ = (int)h_; ww_ = (int)(hh_ / ratio_dst);
    } else {
        // 原图w为较短边，以w为基准截取 h/w 比例的人脸
        ww_ = (int)w_; hh_ = (int)(ww_ * ratio_dst);
    }

    int x1 = (int)box.x1;
    int x2 = (int)box.x2;
    int y1 = (int)box.y1; 
    int y2 = (int)box.y2;
    int xx1 = 0, yy1 = 0;
    int xx2 = ww_ - 1;
    int yy2 = hh_ - 1;
    if (x1 < 0){xx1 = - x1; x1 = 0;}
    if (y1 < 0){yy1 = - y1; y1 = 0;}
    if (x2 > im.w - 1){xx2 = (x2-x1+1) + im.w - x2 - 2; x2 = im.w - 1;}
    if (y2 > im.h - 1){yy2 = (y2-y1+1) + im.h - y2 - 2; y2 = im.h - 1;}
    
    // crop
    image croped = make_image(ww_, hh_, im.c);
    for (int k = 0; k < im.c; k++ ){
        for (int j = yy1; j < yy2 + 1; j++ ){
            for (int i = xx1; i < xx2 + 1; i++ ){
                int x = x1 + i; int y = y1 + j;
                float val = im.data[x + y*im.w + k*im.w*im.h];
                croped.data[i + j*ww_ + k*ww_*hh_] = val;
            }
        }
    }

    *scale = (float)w / (float)ww_;
    image resized = resize_image(croped, w, h);
    free_image(croped);
    return resized;
}

/*
 * @params  
 * -    im:     原图
 * -    box:    回归框
 * -    h, w:   期望得到的图片尺寸
 * @returns
 *      resized: 切割并矫正后的图像
 * @notes 
 */
image image_crop_aligned(image im, bbox box, landmark srcMk, landmark offset, int h, int w, int mode)
{
    float scale = -1.;
    // 以回归框将人脸切出
    image croped = image_crop(im, box, h, w, &scale);
    
    // 计算在剪切出的人脸图像中，关键点的坐标
    float x1 = (box.x1 + box.x2 - croped.w / scale) / 2.;
    float y1 = (box.y1 + box.y2 - croped.h / scale) / 2.;

    landmark dstMk = dstLandmark(offset, croped.h, croped.w);
    srcMk = substract_offset(srcMk, x1, y1);
    srcMk = multiply_scale(srcMk, scale);
    
    // 计算变换矩阵
    CvMat* srcPtMat = landmark_to_cvMat(srcMk);
    CvMat* dstPtMat = landmark_to_cvMat(dstMk);
    CvMat* M = cp2form(srcPtMat, dstPtMat, mode);
    cvReleaseMat(&srcPtMat); cvReleaseMat(&dstPtMat); 

    // 用矩阵变换图像
    IplImage* srcIpl = image_to_ipl(croped);
    IplImage* dstIpl = cvCloneImage(srcIpl);
    cvWarpAffine(srcIpl, dstIpl, M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
    free_image(croped); cvReleaseMat(&M);
    // 返回
    image warped = ipl_to_image(dstIpl);
    cvReleaseImage(&srcIpl); cvReleaseImage(&dstIpl);

    return warped;
}

image image_aligned_v2(image im, landmark src, landmark dst, int h, int w, int mode)
{
    // 计算变换矩阵
    CvMat* srcPtMat = landmark_to_cvMat(src);
    CvMat* dstPtMat = landmark_to_cvMat(dst);
    CvMat* M = cp2form(srcPtMat, dstPtMat, mode);
    cvReleaseMat(&srcPtMat); cvReleaseMat(&dstPtMat); 
    // 变换图像
    IplImage* srcIpl = image_to_ipl(im);
    IplImage* dstIpl = cvCloneImage(srcIpl);
    cvWarpAffine(srcIpl, dstIpl, M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
    // 截取图像
    cvSetImageROI(dstIpl, cvRect(0, 0, w, h));
    IplImage* warpedIpl = cvCreateImage(cvSize(w, h), dstIpl->depth, dstIpl->nChannels);
    cvCopy(dstIpl, warpedIpl, NULL); cvResetImageROI(dstIpl);
    cvReleaseImage(&srcIpl); cvReleaseImage(&dstIpl); cvReleaseMat(&M);

    image warped = ipl_to_image(warpedIpl);
    return warped;
}
