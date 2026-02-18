// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"
#include <opencv2/dnn.hpp>
#include <regex>

// #define benchmark
#define min(a, b) (((a) < (b)) ? (a) : (b))
YOLO_V8::YOLO_V8() {}

YOLO_V8::~YOLO_V8() {
  // Clean up pooled sessions if any
  for (auto s : m_sessionPool) {
    if (s)
      delete s;
  }
  m_sessionPool.clear();
}

#ifdef USE_CUDA
namespace Ort {
template <> struct TypeToTensorType<half> {
  static constexpr ONNXTensorElementDataType type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};
} // namespace Ort
#endif

// BlobFromImage replaced by cv::dnn::blobFromImage in RunSession


char *YOLO_V8::PreProcess(cv::Mat &iImg, std::vector<int> iImgSize,
                          cv::Mat &oImg) {
  // Optimization: Zero-copy resizing and letterboxing
  // We write directly into oImg (which should be m_letterboxBuffer)
  
  int target_h = iImgSize.at(0);
  int target_w = iImgSize.at(1);
  
  // Ensure buffer is allocated
  if (oImg.size() != cv::Size(target_w, target_h) || oImg.type() != CV_8UC3) {
      oImg.create(target_h, target_w, CV_8UC3);
  }
  
  // Fill with padding color (usually 114 for YOLO, but 0 in original code)
  oImg.setTo(cv::Scalar(0, 0, 0));

  switch (modelType) {
  case YOLO_DETECT_V8:
  case YOLO_POSE:
  case YOLO_DETECT_V8_HALF:
  case YOLO_POSE_V8_HALF: // LetterBox
  {
    float r = min(target_w / (float)iImg.cols, target_h / (float)iImg.rows);
    int resized_w = static_cast<int>(iImg.cols * r);
    int resized_h = static_cast<int>(iImg.rows * r);
    resizeScales = 1.0f / r;
    
    // Resize directly into the buffer's ROI (Optimization: Avoids temp allocation)
    // Note: We keep BGR format here and use swapRB=true in blobFromImage later
    cv::resize(iImg, oImg(cv::Rect(0, 0, resized_w, resized_h)), 
               cv::Size(resized_w, resized_h));
    
    break;
  }
  case YOLO_CLS: // CenterCrop
  {
    int h = iImg.rows;
    int w = iImg.cols;
    int m = min(h, w);
    int top = (h - m) / 2;
    int left = (w - m) / 2;
    // Resize directly to output buffer
    cv::resize(iImg(cv::Rect(left, top, m, m)), oImg,
               cv::Size(target_w, target_h));
    break;
  }
  case YOLO_CLS_HALF:
    break;
  }
  return RET_OK;
}

const char *YOLO_V8::CreateSession(DL_INIT_PARAM &iParams) {
  const char *Ret = RET_OK;
  std::regex pattern("[\u4e00-\u9fa5]");
  bool result = std::regex_search(iParams.modelPath, pattern);
  if (result) {
    Ret = "[YOLO_V8]:Your model path is error.Change your model path without "
          "chinese characters.";
    std::cout << Ret << std::endl;
    return Ret;
  }
  try {
    rectConfidenceThreshold = iParams.rectConfidenceThreshold;
    iouThreshold = iParams.iouThreshold;
    imgSize = iParams.imgSize;
    modelType = iParams.modelType;
    cudaEnable = iParams.cudaEnable;
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
    Ort::SessionOptions sessionOption;
    if (iParams.cudaEnable) {
      OrtCUDAProviderOptions cudaOption;
      cudaOption.device_id = 0;
      sessionOption.AppendExecutionProvider_CUDA(cudaOption);
    }

    // Multi-threading optimization - Phase 1
    sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
    sessionOption.SetInterOpNumThreads(iParams.interOpNumThreads);
    sessionOption.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

    // Additional performance optimizations
    sessionOption.SetExecutionMode(ORT_SEQUENTIAL);

#ifdef _WIN32
    // Set process priority for better performance
    HANDLE process = GetCurrentProcess();
    SetPriorityClass(process, HIGH_PRIORITY_CLASS);

    // Set environment variables for Intel CPU optimization
    std::string ompThreads = std::to_string(iParams.intraOpNumThreads);
    SetEnvironmentVariableA("OMP_NUM_THREADS", ompThreads.c_str());
    SetEnvironmentVariableA("KMP_AFFINITY",
                            "granularity=fine,verbose,compact,1,0");
    SetEnvironmentVariableA("KMP_BLOCKTIME", "1");
    SetEnvironmentVariableA("KMP_SETTINGS", "1");
#else
    setenv("OMP_NUM_THREADS", std::to_string(iParams.intraOpNumThreads).c_str(),
           1);
    setenv("KMP_AFFINITY", "granularity=fine,verbose,compact,1,0", 1);
    setenv("KMP_BLOCKTIME", "1", 1);
    setenv("KMP_SETTINGS", "1", 1);
#endif

    // std::cout << "[YOLO_V8]: Creating optimized session with "
    //           << iParams.intraOpNumThreads << " intra-op threads and "
    //           << iParams.interOpNumThreads << " inter-op threads." << std::endl;

#ifdef _WIN32
    int ModelPathSize = MultiByteToWideChar(
        CP_UTF8, 0, iParams.modelPath.c_str(),
        static_cast<int>(iParams.modelPath.length()), nullptr, 0);
    wchar_t *wide_cstr = new wchar_t[ModelPathSize + 1];
    MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(),
                        static_cast<int>(iParams.modelPath.length()), wide_cstr,
                        ModelPathSize);
    wide_cstr[ModelPathSize] = L'\0';
    const wchar_t *modelPath = wide_cstr;
#else
    const char *modelPath = iParams.modelPath.c_str();
#endif // _WIN32

    // Initialize a pool of sessions (Phase 1: session pool for multi-inference)
    Ort::Session *firstSession =
        new Ort::Session(env, modelPath, sessionOption);
    m_sessionPool.push_back(firstSession);
    for (int i = 1; i < iParams.sessionPoolSize; ++i) {
      Ort::Session *s = new Ort::Session(env, modelPath, sessionOption);
      m_sessionPool.push_back(s);
    }
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Session *primary = m_sessionPool.front();
    size_t inputNodesNum = primary->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
      Ort::AllocatedStringPtr input_node_name =
          primary->GetInputNameAllocated(i, allocator);
      char *temp_buf = new char[50];
      strcpy(temp_buf, input_node_name.get());
      inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = primary->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++) {
      Ort::AllocatedStringPtr output_node_name =
          primary->GetOutputNameAllocated(i, allocator);
      char *temp_buf = new char[10];
      strcpy(temp_buf, output_node_name.get());
      outputNodeNames.push_back(temp_buf);
    }
    options = Ort::RunOptions{nullptr};
    WarmUpSession();
    return RET_OK;
  } catch (const std::exception &e) {
    const char *str1 = "[YOLO_V8]:";
    const char *str2 = e.what();
    std::string result = std::string(str1) + std::string(str2);
    char *merged = new char[result.length() + 1];
    std::strcpy(merged, result.c_str());
    std::cout << merged << std::endl;
    delete[] merged;
    return "[YOLO_V8]:Create session failed.";
  }
}

char *YOLO_V8::RunSession(cv::Mat &iImg, std::vector<DL_RESULT> &oResult) {
  clock_t starttime_1 = 0;
#ifdef benchmark
  starttime_1 = clock();
#endif // benchmark

  char *Ret = RET_OK;
  
  // Use member buffer instead of local stack generic variable
  // Optimization: Zero allocation if size matches
  PreProcess(iImg, imgSize, m_letterboxBuffer);
  
  if (modelType < 4) {
    // Optimization: swapRB=true to handle BGR->RGB conversion here
    cv::dnn::blobFromImage(m_letterboxBuffer, m_commonBlob, 1.0 / 255.0, cv::Size(),
                           cv::Scalar(), true, false);
    float *blob = (float *)m_commonBlob.data;
    std::vector<int64_t> inputNodeDims = {1, 3, imgSize.at(0), imgSize.at(1)};
    TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
  } else {
#ifdef USE_CUDA
    // Unified path: Use cv::dnn::blobFromImage (float) -> convert to Half
    cv::dnn::blobFromImage(m_letterboxBuffer, m_commonBlob, 1.0 / 255.0, cv::Size(),
                           cv::Scalar(), true, false);
    // Convert to FP16
    m_commonBlob.convertTo(m_commonBlobHalf, CV_16F);
    
    half *blob = (half *)m_commonBlobHalf.data;
    std::vector<int64_t> inputNodeDims = {1, 3, imgSize.at(0), imgSize.at(1)};
    TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
  }

  return Ret;
}

template <typename N>
char *YOLO_V8::TensorProcess(clock_t &starttime_1, cv::Mat &iImg, N &blob,
                             std::vector<int64_t> &inputNodeDims,
                             std::vector<DL_RESULT> &oResult) {
  Ort::Value inputTensor =
      Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob,
          3 * imgSize.at(0) * imgSize.at(1), inputNodeDims.data(),
          inputNodeDims.size());
#ifdef benchmark
  clock_t starttime_2 = clock();
#endif // benchmark
  size_t poolSize = m_sessionPool.size();
  Ort::Session *sess = m_sessionPool.front();
  if (poolSize > 0) {
    size_t idx = m_sessionIndex.fetch_add(1) % poolSize;
    sess = m_sessionPool[idx];
  }
  auto outputTensor = sess->Run(options, inputNodeNames.data(), &inputTensor, 1,
                                outputNodeNames.data(), outputNodeNames.size());
#ifdef benchmark
  clock_t starttime_3 = clock();
#endif // benchmark

  Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
  auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
  auto output =
      outputTensor.front()
          .GetTensorMutableData<typename std::remove_pointer<N>::type>();
  // delete[] blob; // Optimization: Removed deletion of managed blob
  switch (modelType) {
  case YOLO_DETECT_V8:
  case YOLO_DETECT_V8_HALF: {
    int signalResultNum = outputNodeDims[1]; // 84
    int strideNum = outputNodeDims[2];       // 8400
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rawData;
    if (modelType == YOLO_DETECT_V8) {
      // FP32
      rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
    } else {
      // FP16
      rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
      rawData.convertTo(rawData, CV_32F);
    }
    // Note:
    // ultralytics add transpose operator to the output of yolov8 model.which
    // make yolov8/v5/v7 has same shape
    // https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
    rawData = rawData.t();

    float *data = (float *)rawData.data;

    for (int i = 0; i < strideNum; ++i) {
      float *classesScores = data + 4;
      cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
      cv::Point class_id;
      double maxClassScore;
      cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
      if (maxClassScore > rectConfidenceThreshold) {
        confidences.push_back(maxClassScore);
        class_ids.push_back(class_id.x);
        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];

        int left = int((x - 0.5 * w) * resizeScales);
        int top = int((y - 0.5 * h) * resizeScales);

        int width = int(w * resizeScales);
        int height = int(h * resizeScales);

        boxes.push_back(cv::Rect(left, top, width, height));
      }
      data += signalResultNum;
    }
    // Draw detections removed. Rendering handled by QML.
    
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold,
                      nmsResult);
    for (int i = 0; i < nmsResult.size(); ++i) {
      int idx = nmsResult[i];
      DL_RESULT result;
      result.classId = class_ids[idx];
      result.confidence = confidences[idx];
      result.box = boxes[idx];
      oResult.push_back(result);
    }

#ifdef benchmark
    clock_t starttime_4 = clock();
    double pre_process_time =
        (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
    double process_time =
        (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
    double post_process_time =
        (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
    if (cudaEnable) {
      std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, "
                << process_time << "ms inference, " << post_process_time
                << "ms post-process." << std::endl;
    } else {
      std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, "
                << process_time << "ms inference, " << post_process_time
                << "ms post-process." << std::endl;
    }
#endif // benchmark

    break;
  }
  case YOLO_CLS:
  case YOLO_CLS_HALF: {
    cv::Mat rawData;
    if (modelType == YOLO_CLS) {
      // FP32
      rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
    } else {
      // FP16
      rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
      rawData.convertTo(rawData, CV_32F);
    }
    float *data = (float *)rawData.data;

    DL_RESULT result;
    for (int i = 0; i < this->classes.size(); i++) {
      result.classId = i;
      result.confidence = data[i];
      oResult.push_back(result);
    }
    break;
  }
  default:
    std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
  }
  return RET_OK;
}

char *YOLO_V8::WarmUpSession() {
  clock_t starttime_1 = clock();
  cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(1), imgSize.at(0)), CV_8UC3);
  cv::Mat processedImg;
  PreProcess(iImg, imgSize, processedImg);
  if (modelType < 4) {
    for (auto s : m_sessionPool) {
      cv::Mat blobMat;
      cv::dnn::blobFromImage(processedImg, blobMat, 1.0 / 255.0, cv::Size(),
                             cv::Scalar(), false, false);
      float *blob = (float *)blobMat.data;
      std::vector<int64_t> YOLO_input_node_dims = {1, 3, imgSize.at(0),
                                                   imgSize.at(1)};
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob,
          3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(),
          YOLO_input_node_dims.size());
      auto output_tensors =
          s->Run(options, inputNodeNames.data(), &input_tensor, 1,
                 outputNodeNames.data(), outputNodeNames.size());
      // delete[] blob; // No need to delete, managed by cv::Mat
      clock_t starttime_4 = clock();
      double post_process_time =
          (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
      if (cudaEnable) {
        std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost "
                  << post_process_time << " ms. " << std::endl;
      } else {
        // std::cout << "[YOLO_V8(CPU)]: " << "Warm-up completed in "
        //           << post_process_time << " ms. " << std::endl;
      }
    }
  } else {
#ifdef USE_CUDA
    for (auto s : m_sessionPool) {
      cv::Mat blobMat;
      cv::dnn::blobFromImage(processedImg, blobMat, 1.0 / 255.0, cv::Size(),
                             cv::Scalar(), false, false);
      cv::Mat blobMatHalf;
      blobMat.convertTo(blobMatHalf, CV_16F);
      half *blob = (half *)blobMatHalf.data;
      std::vector<int64_t> YOLO_input_node_dims = {1, 3, imgSize.at(0),
                                                   imgSize.at(1)};
      Ort::Value input_tensor = Ort::Value::CreateTensor<half>(
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob,
          3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(),
          YOLO_input_node_dims.size());
      auto output_tensors =
          s->Run(options, inputNodeNames.data(), &input_tensor, 1,
                 outputNodeNames.data(), outputNodeNames.size());
      // delete[] blob; // Managed by cv::Mat
      clock_t starttime_4 = clock();
      double post_process_time =
          (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
      if (cudaEnable) {
        std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost "
                  << post_process_time << " ms. " << std::endl;
      }
    }
#endif
  }
  return RET_OK;
}
