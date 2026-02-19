// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#define RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include "onnxruntime_cxx_api.h"
#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

enum MODEL_TYPE {
  // FLOAT32 MODEL
  YOLO_DETECT_V8 = 1,
  YOLO_POSE = 2,
  YOLO_CLS = 3,

  // FLOAT16 MODEL
  YOLO_DETECT_V8_HALF = 4,
  YOLO_POSE_V8_HALF = 5,
  YOLO_CLS_HALF = 6
};

typedef struct _DL_INIT_PARAM {
  std::string modelPath;
  MODEL_TYPE modelType = YOLO_DETECT_V8;
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
} DL_RESULT;

class YOLO_V8 {
public:
  YOLO_V8();

  ~YOLO_V8();

public:
  const char *CreateSession(DL_INIT_PARAM &iParams);

  typedef struct _InferenceTiming {
    double preProcessTime;
    double inferenceTime;
    double postProcessTime;
  } InferenceTiming;

  char *RunSession(const cv::Mat &iImg, std::vector<DL_RESULT> &oResult, InferenceTiming &timing);

  char *WarmUpSession();

  template <typename N>
  char *TensorProcess(std::chrono::high_resolution_clock::time_point &start_pre, const cv::Mat &iImg, N &blob,
                      std::vector<int64_t> &inputNodeDims,
                      std::vector<DL_RESULT> &oResult, InferenceTiming &timing);

  char *PreProcess(const cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg);

  std::vector<std::string> classes{};

private:
  Ort::Env env;
  // Session pool for multi-session inference (Phase 1 completion)
  std::vector<Ort::Session *> m_sessionPool; // owned sessions
  // Shared input/output node names for all sessions (assumes IO is identical
  // across sessions)
  std::vector<const char *> inputNodeNames;
  std::vector<const char *> outputNodeNames;
  bool cudaEnable;
  Ort::RunOptions options;
  // Input/output names are shared across sessions to avoid duplicating storage

  MODEL_TYPE modelType;
  std::vector<int> imgSize;
  float rectConfidenceThreshold;
  float iouThreshold;
  float resizeScales; // letterbox scale
  std::atomic<size_t> m_sessionIndex{0};
  
  // Optimization: Reusable memory for blob to avoid reallocations
  cv::Mat m_commonBlob; 
  cv::Mat m_commonBlobHalf;
  cv::Mat m_letterboxBuffer;
};
