#include "preprocess.h"

ImagePreProcessor::ImagePreProcessor(MODEL_TYPE modelType, const std::vector<int>& imgSize)
    : modelType(modelType), imgSize(imgSize), resizeScales(1.0f) {}

char *ImagePreProcessor::PreProcess(const cv::Mat &iImg, cv::Mat &oImg) {
    int target_h = imgSize.at(0);
    int target_w = imgSize.at(1);
    
    float r = std::min(target_w / (float)iImg.cols, target_h / (float)iImg.rows);
    int resized_w = static_cast<int>(iImg.cols * r);
    int resized_h = static_cast<int>(iImg.rows * r);
    resizeScales = 1.0f / r;

    if (oImg.size() != cv::Size(target_w, target_h) || oImg.type() != CV_8UC3) {
        oImg.create(target_h, target_w, CV_8UC3);
    }
    oImg.setTo(cv::Scalar(114, 114, 114)); // YOLO padding color

    cv::Mat resized;
    cv::resize(iImg, resized, cv::Size(resized_w, resized_h));
    resized.copyTo(oImg(cv::Rect(0, 0, resized_w, resized_h)));
    
    return RET_OK;
}

void ImagePreProcessor::PreProcessImageToBlob(const cv::Mat& iImg, float* blob_data) {
    // 1-2ms bottleneck was likely the manual pixel loop.
    // Using OpenCV's split + convertTo is multi-threaded and extremely fast.
    int w = iImg.cols;
    int h = iImg.rows;
    
    cv::Mat channels[3];
    cv::split(iImg, channels); // B, G, R
    
    // Map the float* buffer to 3 single-channel Mats (planes)
    // YOLO expects RGB layout
    cv::Mat pR(h, w, CV_32FC1, blob_data);
    cv::Mat pG(h, w, CV_32FC1, blob_data + w * h);
    cv::Mat pB(h, w, CV_32FC1, blob_data + 2 * w * h);
    
    channels[2].convertTo(pR, CV_32FC1, 1.0 / 255.0);
    channels[1].convertTo(pG, CV_32FC1, 1.0 / 255.0);
    channels[0].convertTo(pB, CV_32FC1, 1.0 / 255.0);
}
