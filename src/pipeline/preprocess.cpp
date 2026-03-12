#include "preprocess.h"
#include "simd_utils.h"

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

    // Resize directly into the ROI of the destination image
    cv::Mat roi = oImg(cv::Rect(0, 0, resized_w, resized_h));
    cv::resize(iImg, roi, cv::Size(resized_w, resized_h));
    
    return RET_OK;
}

void ImagePreProcessor::PreProcessImageToBlob(const cv::Mat& iImg, float* blob_data) {
    // Optimized single-pass fused loop for HWC -> CHW + BGR -> RGB + Normalization
    // This replaces cv::split + 3x convertTo which requires 4 passes over data.
    simd::hwc_to_chw_bgr_to_rgb_sse41(iImg.data, blob_data, iImg.cols, iImg.rows, iImg.step);
}
