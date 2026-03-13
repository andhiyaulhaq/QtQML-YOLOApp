// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include "YoloTypes.h"
#include <memory>

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

// Forward declarations
class IInferenceBackend;
class ImagePreProcessor;
class IPostProcessor;

class YoloPipeline {
public:
  YoloPipeline();
  ~YoloPipeline();

public:
  const char *CreateSession(DL_INIT_PARAM &iParams);

  typedef struct _InferenceTiming {
    double preProcessTime;
    double inferenceTime;
    double postProcessTime;
  } InferenceTiming;

  char *RunSession(const cv::Mat &iImg, std::vector<DL_RESULT> &oResult, InferenceTiming &timing);
  char *WarmUpSession();

  const std::vector<std::string>& getClassNames() const { return classes; }

  std::vector<std::string> classes{};

private:
  std::unique_ptr<IInferenceBackend> m_backend;
  
  MODEL_TYPE modelType;
  std::vector<int> imgSize;
  
  // Optimization: Reusable memory for blob to avoid reallocations
  cv::Mat m_commonBlob; 
  cv::Mat m_letterboxBuffer;

  std::unique_ptr<ImagePreProcessor> m_preProcessor;
  std::unique_ptr<IPostProcessor> m_postProcessor;
};
