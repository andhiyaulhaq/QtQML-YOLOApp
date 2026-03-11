#pragma once

#define RET_OK nullptr

#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <opencv2/opencv.hpp>

enum MODEL_TYPE {
  YOLO_DETECT = 1,
  YOLO_POSE = 2,
  YOLO_SEG = 3
};

typedef struct _DL_INIT_PARAM {
  std::string modelPath;
  MODEL_TYPE modelType = YOLO_DETECT;
  std::vector<int> imgSize = {640, 640};
  float rectConfidenceThreshold = 0.4;
  float iouThreshold = 0.5;
  int keyPointsNum = 2; // Note:kpt number for pose
  bool cudaEnable = false;
  int logSeverityLevel = 3;
  int intraOpNumThreads = std::max(1u, std::thread::hardware_concurrency() / 2);
  int interOpNumThreads = 1;
  int sessionPoolSize = 1;
} DL_INIT_PARAM;

typedef struct _DL_RESULT {
  int classId;
  float confidence;
  cv::Rect box;
  std::vector<cv::Point2f> keyPoints;
  cv::Mat boxMask;
} DL_RESULT;
