#include "util_cxx.h"

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

void saveFloatArray(const String filename, float *X, int n)
{
    FILE* fp = fopen(filename.c_str(), "wb");
    fwrite(X, sizeof(float), n, fp);
    fclose(fp);
}

void loadFloatArray(const String filename, float *X, int n)
{
    FILE* fp = fopen(filename.c_str(), "rb+");
    fread(X, sizeof(float), n, fp);
    fclose(fp);
}
