#include "YoloPipeline.h"
#include "PreProcessor.h"
#include "PostProcessor.h"
#include "backends/OnnxRuntimeBackend.h"
#include "backends/OpenVinoBackend.h"
#include <regex>

YoloPipeline::YoloPipeline() {}

YoloPipeline::~YoloPipeline() {}

const char *YoloPipeline::CreateSession(DL_INIT_PARAM &iParams) {
    const char *Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    if (std::regex_search(iParams.modelPath, pattern)) {
        return "[YoloPipeline]: Model path cannot contain Chinese characters.";
    }

    try {
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;

        m_preProcessor = std::make_unique<ImagePreProcessor>(modelType, imgSize);

        switch (modelType) {
            case YOLO_DETECT:
                m_postProcessor = std::make_unique<DetectionPostProcessor>(modelType, iParams.rectConfidenceThreshold, iParams.iouThreshold);
                break;
            case YOLO_POSE:
                m_postProcessor = std::make_unique<PosePostProcessor>(modelType, iParams.rectConfidenceThreshold, iParams.iouThreshold);
                break;
            case YOLO_SEG:
                m_postProcessor = std::make_unique<SegmentationPostProcessor>(modelType, iParams.rectConfidenceThreshold, iParams.iouThreshold);
                break;
            default:
                throw std::runtime_error("Unsupported model type.");
        }

        // Strategy Pattern: Instantiate backend
        if (iParams.runtimeType == RUNTIME_ONNXRUNTIME) {
            m_backend = std::make_unique<OnnxRuntimeBackend>();
        } else {
            m_backend = std::make_unique<OpenVinoBackend>();
        }

        const char* backendStatus = m_backend->createSession(iParams);
        if (backendStatus != RET_OK) return backendStatus;

        // Initialize post-processor buffers
        std::vector<int64_t> outShape = m_backend->getOutputShape();
        if (!outShape.empty() && outShape.size() >= 3) {
            m_postProcessor->initBuffers(static_cast<size_t>(outShape[2]));
        } else {
            m_postProcessor->initBuffers(8400); // Fallback
        }

        WarmUpSession();
        return RET_OK;
    } catch (const std::exception &e) {
        std::cerr << "[YoloPipeline]: " << e.what() << std::endl;
        return "[YoloPipeline]: Create session failed.";
    }
}

char *YoloPipeline::RunSession(const cv::Mat &iImg, std::vector<DL_RESULT> &oResult, InferenceTiming &timing) {
    auto start_pre = std::chrono::high_resolution_clock::now();

    // Step 1: Letterbox Resize (CPU)
    LetterboxInfo info = m_preProcessor->PreProcess(iImg, m_letterboxBuffer);

    int height = imgSize.at(0);
    int width = imgSize.at(1);
    
    // Step 2: NCHW + Normalization (Optimized)
    int sz[] = {1, 3, height, width};
    m_commonBlob.create(4, sz, CV_32F);
    float* blob_data = m_commonBlob.ptr<float>();
    m_preProcessor->PreProcessImageToBlob(m_letterboxBuffer, blob_data);

    std::vector<int64_t> inputNodeDims = {1, 3, (int64_t)height, (int64_t)width};
    auto end_pre = std::chrono::high_resolution_clock::now();
    timing.preProcessTime = std::chrono::duration<double, std::milli>(end_pre - start_pre).count();

    // Inference
    auto start_infer = std::chrono::high_resolution_clock::now();
    InferenceOutput out = m_backend->runInference(blob_data, inputNodeDims);
    auto end_infer = std::chrono::high_resolution_clock::now();
    timing.inferenceTime = std::chrono::duration<double, std::milli>(end_infer - start_infer).count();

    // Post-processing
    auto start_post = std::chrono::high_resolution_clock::now();
    m_postProcessor->PostProcess(out.primaryData, out.primaryShape, oResult, 
                                 info, classes, 
                                 out.secondaryData, out.secondaryShape);
    auto end_post = std::chrono::high_resolution_clock::now();
    timing.postProcessTime = std::chrono::duration<double, std::milli>(end_post - start_post).count();

    return RET_OK;
}

char *YoloPipeline::WarmUpSession() {
    if (m_backend) {
        m_backend->warmUp(imgSize);
    }
    return RET_OK;
}
