// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include "yolo_types.h"
#include <memory>

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include "onnxruntime_cxx_api.h"
#include <atomic>

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

class ImagePreProcessor;
class IPostProcessor;

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



  const std::vector<std::string>& getClassNames() const { return classes; }

  std::vector<std::string> classes{};

private:
  Ort::Env env;
  // Session pool for multi-session inference (Phase 1 completion)
  std::vector<Ort::Session *> m_sessionPool; // owned sessions
  // Shared input/output node names for all sessions (assumes IO is identical
  // across sessions)
  std::vector<const char *> inputNodeNames;
  std::vector<const char *> outputNodeNames;
  std::vector<std::string> m_inputNodeNameStorage;
  std::vector<std::string> m_outputNodeNameStorage;
  bool cudaEnable;
  Ort::RunOptions options;
  // Input/output names are shared across sessions to avoid duplicating storage

  MODEL_TYPE modelType;
  std::vector<int> imgSize;
  std::atomic<size_t> m_sessionIndex{0};
  
  // Optimization: Reusable memory for blob to avoid reallocations
  cv::Mat m_commonBlob; 
  cv::Mat m_commonBlobHalf;
  cv::Mat m_letterboxBuffer;

  std::unique_ptr<ImagePreProcessor> m_preProcessor;
  std::unique_ptr<IPostProcessor> m_postProcessor;
};
