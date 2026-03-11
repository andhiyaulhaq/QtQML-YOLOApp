#include "preprocess.h"

ImagePreProcessor::ImagePreProcessor(MODEL_TYPE modelType, const std::vector<int>& imgSize)
    : modelType(modelType), imgSize(imgSize), resizeScales(1.0f) {}

char *ImagePreProcessor::PreProcess(const cv::Mat &iImg, cv::Mat &oImg) {
  int target_h = imgSize.at(0);
  int target_w = imgSize.at(1);
  
  if (oImg.size() != cv::Size(target_w, target_h) || oImg.type() != CV_8UC3) {
      oImg.create(target_h, target_w, CV_8UC3);
  }
  
  oImg.setTo(cv::Scalar(0, 0, 0));

  switch (modelType) {
  case YOLO_DETECT:
  case YOLO_POSE:
  case YOLO_SEG:
  {
    float r = std::min(target_w / (float)iImg.cols, target_h / (float)iImg.rows);
    int resized_w = static_cast<int>(iImg.cols * r);
    int resized_h = static_cast<int>(iImg.rows * r);
    resizeScales = 1.0f / r;
    
    cv::resize(iImg, oImg(cv::Rect(0, 0, resized_w, resized_h)), 
               cv::Size(resized_w, resized_h));
    
    break;
  }
  }
  return RET_OK;
}

void ImagePreProcessor::PreProcessImageToBlob(const cv::Mat& iImg, float* blob_data) {
    int channels = 3;
    int height = imgSize.at(0);
    int width = imgSize.at(1);
    
    const uint8_t* img_data = iImg.data;
    int step = width * channels; 

    int plane_0 = 0;                  
    int plane_1 = height * width;     
    int plane_2 = 2 * height * width; 
    
    for (int h = 0; h < height; ++h) {
        const uint8_t* row_ptr = img_data + h * iImg.step;
        for (int w = 0; w < width; ++w) {
            uint8_t b = row_ptr[w * 3 + 0];
            uint8_t g = row_ptr[w * 3 + 1];
            uint8_t r = row_ptr[w * 3 + 2];

            int offset = h * width + w;
            
            blob_data[plane_0 + offset] = r / 255.0f;
            blob_data[plane_1 + offset] = g / 255.0f;
            blob_data[plane_2 + offset] = b / 255.0f;
        }
    }
}
