// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"
#include <opencv2/dnn.hpp>
#include <regex>

#include <numeric>
#include <algorithm>

// #define benchmark
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


char *YOLO_V8::PreProcess(const cv::Mat &iImg, std::vector<int> iImgSize,
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
    float r = std::min(target_w / (float)iImg.cols, target_h / (float)iImg.rows);
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
    int m = std::min(h, w);
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
    // Set environment variables for Intel CPU optimization
    std::string ompThreads = std::to_string(iParams.intraOpNumThreads);
    SetEnvironmentVariableA("OMP_NUM_THREADS", ompThreads.c_str());
    SetEnvironmentVariableA("KMP_AFFINITY", "granularity=fine,verbose,compact,1,0");
    SetEnvironmentVariableA("KMP_BLOCKTIME", "1");
    SetEnvironmentVariableA("KMP_SETTINGS", "1");
#else
    setenv("OMP_NUM_THREADS", std::to_string(iParams.intraOpNumThreads).c_str(), 1);
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
    try {
        Ort::Session *firstSession = new Ort::Session(env, modelPath, sessionOption);
        m_sessionPool.push_back(firstSession);
    } catch (const std::exception &e) {
        if (iParams.cudaEnable) {
            std::cout << "[YOLO_V8]: CUDA initialization failed (" << e.what() << "). Falling back to CPU." << std::endl;
            // Fallback: Disable CUDA and try again
            iParams.cudaEnable = false;
            cudaEnable = false;
            // Re-create session options without CUDA
            sessionOption = Ort::SessionOptions();
            sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
            sessionOption.SetInterOpNumThreads(iParams.interOpNumThreads);
            sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);
            sessionOption.SetExecutionMode(ORT_SEQUENTIAL);
            
            Ort::Session *firstSession = new Ort::Session(env, modelPath, sessionOption);
            m_sessionPool.push_back(firstSession);
        } else {
            throw e; // Re-throw if CPU also failed
        }
    }

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
      m_inputNodeNameStorage.push_back(std::string(input_node_name.get()));
    }
    for (const auto& name : m_inputNodeNameStorage) {
      inputNodeNames.push_back(name.c_str());
    }
    size_t OutputNodesNum = primary->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++) {
      Ort::AllocatedStringPtr output_node_name =
          primary->GetOutputNameAllocated(i, allocator);
      m_outputNodeNameStorage.push_back(std::string(output_node_name.get()));
    }
    for (const auto& name : m_outputNodeNameStorage) {
      outputNodeNames.push_back(name.c_str());
    }
    options = Ort::RunOptions{nullptr};
    WarmUpSession();

    // Pre-allocate Two-Pass postprocessing buffers
    {
        int strideNum = 8400;
        Ort::TypeInfo typeInfo = m_sessionPool.front()->GetOutputTypeInfo(0);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        auto shape = tensorInfo.GetShape();
        if (shape.size() >= 3 && shape[2] > 0) {
            strideNum = static_cast<int>(shape[2]);
        }
        m_bestScores.resize(strideNum);
        m_bestClassIds.resize(strideNum);
        m_classIds.reserve(256);
        m_confidences.reserve(256);
        m_boxes.reserve(256);
        m_nmsIndices.reserve(64);
        m_sortIndices.reserve(256);
    }

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

char *YOLO_V8::RunSession(const cv::Mat &iImg, std::vector<DL_RESULT> &oResult, InferenceTiming &timing) {
  auto start_pre = std::chrono::high_resolution_clock::now();

  char *Ret = RET_OK;

  // Use member buffer instead of local stack generic variable
  // Optimization: Zero allocation if size matches
  PreProcess(iImg, imgSize, m_letterboxBuffer);

  // Manual pre-processing (replacing cv::dnn::blobFromImage)
  // 1. Ensure output blob buffer is allocated with correct dimensions (NCHW)
  int channels = 3;
  int height = imgSize.at(0);
  int width = imgSize.at(1);
  
  if (modelType < 4) { // FLOAT32 models
      // Ensure m_commonBlob is NCHW [1, 3, h, w] CV_32F
      // OpenCV create() reuses buffer if size/type matches
      int sz[] = {1, channels, height, width};
      m_commonBlob.create(4, sz, CV_32F);
      
      float* blob_data = m_commonBlob.ptr<float>();
      const uint8_t* img_data = m_letterboxBuffer.data;
      int step = width * channels; // bytes per row in source

      // Manual loop: BGR -> RGB, /255.0, HWC -> CHW
      // This is the "Hot Loop" for preprocessing
      // Planar offsets
      int plane_0 = 0;                  // R channel (dst)
      int plane_1 = height * width;     // G channel (dst)
      int plane_2 = 2 * height * width; // B channel (dst)
      
      for (int h = 0; h < height; ++h) {
          const uint8_t* row_ptr = img_data + h * m_letterboxBuffer.step;
          for (int w = 0; w < width; ++w) {
              // Source is BGR (OpenCV default)
              uint8_t b = row_ptr[w * 3 + 0];
              uint8_t g = row_ptr[w * 3 + 1];
              uint8_t r = row_ptr[w * 3 + 2];

              // Dest is RGB Planar (CHW) + Normalize
              // Access: blob_data[channel * H * W + y * W + x]
              int offset = h * width + w;
              
              blob_data[plane_0 + offset] = r / 255.0f;
              blob_data[plane_1 + offset] = g / 255.0f;
              blob_data[plane_2 + offset] = b / 255.0f;
          }
      }

      std::vector<int64_t> inputNodeDims = {1, 3, height, width};
      auto end_pre = std::chrono::high_resolution_clock::now();
      timing.preProcessTime = std::chrono::duration<double, std::milli>(end_pre - start_pre).count();

      TensorProcess(start_pre, iImg, blob_data, inputNodeDims, oResult, timing); 

  } else { // FLOAT16 models (reuse same logic, convert at end or use half ptr)
#ifdef USE_CUDA
      // For simplicity in this phase, we do the same float conversion then convert to half
      // A full optimization would write directly to half, but requires half-float logic
      
      int sz[] = {1, channels, height, width};
      m_commonBlob.create(4, sz, CV_32F);
      
      float* blob_data = m_commonBlob.ptr<float>();
      const uint8_t* img_data = m_letterboxBuffer.data;
      
      int plane_0 = 0;
      int plane_1 = height * width;
      int plane_2 = 2 * height * width;
      
       for (int h = 0; h < height; ++h) {
          const uint8_t* row_ptr = img_data + h * m_letterboxBuffer.step;
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

      // Convert to FP16
      m_commonBlob.convertTo(m_commonBlobHalf, CV_16F);
      half *blob = (half *)m_commonBlobHalf.data;
      std::vector<int64_t> inputNodeDims = {1, 3, height, width};

      auto end_pre = std::chrono::high_resolution_clock::now();
      timing.preProcessTime = std::chrono::duration<double, std::milli>(end_pre - start_pre).count();
      
      TensorProcess(start_pre, iImg, blob, inputNodeDims, oResult, timing);
#endif
  }

  return Ret;
}

template <typename N>
char *YOLO_V8::TensorProcess(std::chrono::high_resolution_clock::time_point &start_pre, const cv::Mat &iImg, N &blob,
                             std::vector<int64_t> &inputNodeDims,
                             std::vector<DL_RESULT> &oResult, InferenceTiming &timing) {
  Ort::Value inputTensor =
      Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob,
          3 * imgSize.at(0) * imgSize.at(1), inputNodeDims.data(),
          inputNodeDims.size());

  auto start_infer = std::chrono::high_resolution_clock::now();
  
  size_t poolSize = m_sessionPool.size();
  Ort::Session *sess = m_sessionPool.front();
  if (poolSize > 0) {
    size_t idx = m_sessionIndex.fetch_add(1) % poolSize;
    sess = m_sessionPool[idx];
  }
  auto outputTensor = sess->Run(options, inputNodeNames.data(), &inputTensor, 1,
                                outputNodeNames.data(), outputNodeNames.size());

  auto end_infer = std::chrono::high_resolution_clock::now();
  timing.inferenceTime = std::chrono::duration<double, std::milli>(end_infer - start_infer).count();
  
  auto start_post = std::chrono::high_resolution_clock::now();

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
    // ──────────────────────────────────────────────────────────
    // DIMENSIONS
    // ──────────────────────────────────────────────────────────
    int signalResultNum = outputNodeDims[1]; // 84
    int strideNum = outputNodeDims[2];       // 8400
    int numClasses = signalResultNum - 4;    // 80

    // ──────────────────────────────────────────────────────────
    // DATA POINTER (Zero-copy for FP32, conversion for FP16)
    // ──────────────────────────────────────────────────────────
    float* data;
    cv::Mat rawData; // Only used for FP16 conversion; empty for FP32
    if (modelType == YOLO_DETECT_V8) {
        data = static_cast<float*>(output);  // Direct pointer — zero allocation
    } else {
        rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
        rawData.convertTo(rawData, CV_32F);  // FP16 → FP32 (allocates once)
        data = reinterpret_cast<float*>(rawData.data);
    }

    // ══════════════════════════════════════════════════════════
    // PASS 1: ROW-MAJOR SCORE SWEEP
    // ══════════════════════════════════════════════════════════
    // ── Step 1A: Initialize with class 0 ──
    float* row0 = data + 4 * strideNum;
    memcpy(m_bestScores.data(), row0, strideNum * sizeof(float));
    memset(m_bestClassIds.data(), 0, strideNum * sizeof(int));

    // ── Step 1B: Sweep classes 1..79 ──
    for (int c = 1; c < numClasses; ++c) {
        const float* rowC = data + (4 + c) * strideNum;
        float* bestS = m_bestScores.data();
        int*   bestC = m_bestClassIds.data();

        // Inner loop: 8400 contiguous float comparisons (auto-vectorizes)
        for (int j = 0; j < strideNum; ++j) {
            if (rowC[j] > bestS[j]) {
                bestS[j] = rowC[j];
                bestC[j] = c;
            }
        }
    }

    // ══════════════════════════════════════════════════════════
    // PASS 2: THRESHOLD + BOX DECODE
    // ══════════════════════════════════════════════════════════
    m_classIds.clear();
    m_confidences.clear();
    m_boxes.clear();

    const float* bestS = m_bestScores.data();
    const int*   bestC = m_bestClassIds.data();

    // Iterate through best scores to find survivors
    for (int j = 0; j < strideNum; ++j) {
        if (bestS[j] > rectConfidenceThreshold) {
            m_confidences.push_back(bestS[j]);
            m_classIds.push_back(bestC[j]);

            // Decode coords
            float cx = data[0 * strideNum + j];
            float cy = data[1 * strideNum + j];
            float bw = data[2 * strideNum + j];
            float bh = data[3 * strideNum + j];

            int left   = static_cast<int>((cx - 0.5f * bw) * resizeScales);
            int top    = static_cast<int>((cy - 0.5f * bh) * resizeScales);
            int width  = static_cast<int>(bw * resizeScales);
            int height = static_cast<int>(bh * resizeScales);

            m_boxes.emplace_back(left, top, width, height);
        }
    }

    // ══════════════════════════════════════════════════════════
    // PASS 3: NON-MAXIMUM SUPPRESSION
    // ══════════════════════════════════════════════════════════
    m_nmsIndices.clear();
    greedyNMS(iouThreshold);

    // ══════════════════════════════════════════════════════════
    // RESULT ASSEMBLY
    // ══════════════════════════════════════════════════════════
    for (size_t i = 0; i < m_nmsIndices.size(); ++i) {
        int idx = m_nmsIndices[i];
        DL_RESULT result;
        result.classId    = m_classIds[idx];
        result.confidence = m_confidences[idx];
        result.box        = m_boxes[idx];
        oResult.push_back(result);
    }

    auto end_post = std::chrono::high_resolution_clock::now();
    timing.postProcessTime = std::chrono::duration<double, std::milli>(end_post - start_post).count();

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

void YOLO_V8::greedyNMS(float iouThresh) {
    // ─────────────────────────────────────────────────
    // Greedy NMS — operates on m_confidences, m_boxes
    // Outputs to m_nmsIndices
    // ─────────────────────────────────────────────────
    
    int n = static_cast<int>(m_confidences.size());
    if (n == 0) return;
    
    // Step 1: Build confidence-sorted index
    // ALLOCATION: m_sortIndices.resize() is O(1) if n <= previous capacity.
    m_sortIndices.resize(n);
    std::iota(m_sortIndices.begin(), m_sortIndices.end(), 0);
    std::sort(m_sortIndices.begin(), m_sortIndices.end(),
              [this](int a, int b) {
                  return m_confidences[a] > m_confidences[b];
              });
    
    // Step 2: Greedy suppression
    m_suppressed.assign(n, false);
    
    for (int i = 0; i < n; ++i) {
        int idx = m_sortIndices[i];
        if (m_suppressed[idx]) continue;    // Already suppressed
        
        m_nmsIndices.push_back(idx);        // This box survives
        
        const cv::Rect& a = m_boxes[idx];
        float areaA = static_cast<float>(a.width * a.height);
        
        // Check all remaining lower-confidence boxes
        for (int k = i + 1; k < n; ++k) {
            int kidx = m_sortIndices[k];
            if (m_suppressed[kidx]) continue;
            
            const cv::Rect& b = m_boxes[kidx];
            
            // ── Quick rejection ──
            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.width,  b.x + b.width);
            int y2 = std::min(a.y + a.height, b.y + b.height);
            
            if (x2 <= x1 || y2 <= y1) continue;  // No overlap → skip
            
            // ── IoU computation ──
            float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
            float areaB = static_cast<float>(b.width * b.height);
            float unionArea = areaA + areaB - intersection;
            float iou = intersection / unionArea;
            
            if (iou > iouThresh) {
                m_suppressed[kidx] = true;  // Suppress this box
            }
        }
    }
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
