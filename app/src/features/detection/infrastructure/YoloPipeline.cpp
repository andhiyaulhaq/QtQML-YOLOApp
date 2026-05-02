#include "YoloPipeline.h"
#include <regex>
#include <chrono>
#include <iostream>
#include <fstream>
#include "backends/OnnxRuntimeBackend.h"
#include "backends/OpenVinoBackend.h"
#include <QDebug>

YoloPipeline::YoloPipeline() {}

YoloPipeline::~YoloPipeline() {}

const char* YoloPipeline::createSession(const InferenceConfig& config) {
    std::regex pattern("[\u4e00-\u9fa5]");
    if (std::regex_search(config.modelPath, pattern)) {
        return "[YoloPipeline]: Model path cannot contain Chinese characters.";
    }

    try {
        m_imgSize = config.imgSize;
        m_taskType = config.taskType;
        qDebug() << "[YoloPipeline]: Initializing with task" << (int)m_taskType;

        m_classes.clear();
        std::ifstream file("assets/classes.txt");
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                m_classes.push_back(line);
            }
        } else {
            qDebug() << "[YoloPipeline]: Warning: Could not open assets/classes.txt. Class names will be empty.";
        }

        m_preProcessor = std::make_unique<ImagePreProcessor>(m_taskType, m_imgSize);

        switch (m_taskType) {
            case YoloTask::TaskType::ObjectDetection:
                m_postProcessor = std::make_unique<DetectionPostProcessor>(m_taskType, config.confidenceThreshold, config.iouThreshold);
                break;
            case YoloTask::TaskType::PoseEstimation:
                m_postProcessor = std::make_unique<PosePostProcessor>(m_taskType, config.confidenceThreshold, config.iouThreshold);
                break;
            case YoloTask::TaskType::ImageSegmentation:
                m_postProcessor = std::make_unique<SegmentationPostProcessor>(m_taskType, config.confidenceThreshold, config.iouThreshold);
                break;
            default:
                throw std::runtime_error("Unsupported task type.");
        }

        if (config.runtimeType == YoloTask::RuntimeType::ONNXRuntime) {
            m_backend = std::make_unique<OnnxRuntimeBackend>();
        } else {
            m_backend = std::make_unique<OpenVinoBackend>();
        }

        const char* backendStatus = m_backend->createSession(config);
        if (backendStatus != nullptr) return backendStatus;

        std::vector<int64_t> outShape = m_backend->getOutputShape();
        if (!outShape.empty() && outShape.size() >= 3) {
            m_postProcessor->initBuffers(static_cast<size_t>(outShape[2]));
        } else {
            m_postProcessor->initBuffers(8400); 
        }

        warmUp();
        return nullptr; // OK
    } catch (const std::exception &e) {
        std::cerr << "[YoloPipeline]: " << e.what() << std::endl;
        return "[YoloPipeline]: Create session failed.";
    }
}

char* YoloPipeline::runInference(const cv::Mat& frame,
                                 std::vector<DetectionResult>& results,
                                 InferenceTiming& timing) {
    qDebug() << "[YoloPipeline]: runInference starting pre-process";
    auto start_pre = std::chrono::high_resolution_clock::now();

    LetterboxInfo info = m_preProcessor->preProcess(frame, m_letterboxBuffer);

    int height = m_imgSize.at(0);
    int width = m_imgSize.at(1);
    
    int sz[] = {1, 3, height, width};
    m_commonBlob.create(4, sz, CV_32F);
    float* blob_data = m_commonBlob.ptr<float>();
    m_preProcessor->preProcessImageToBlob(m_letterboxBuffer, blob_data);

    std::vector<int64_t> inputNodeDims = {1, 3, (int64_t)height, (int64_t)width};
    auto end_pre = std::chrono::high_resolution_clock::now();
    timing.preProcess = std::chrono::duration<double, std::milli>(end_pre - start_pre).count();

    qDebug() << "[YoloPipeline]: pre-process done, starting backend inference";
    auto start_infer = std::chrono::high_resolution_clock::now();
    InferenceOutput out = m_backend->runInference(blob_data, inputNodeDims);
    auto end_infer = std::chrono::high_resolution_clock::now();
    timing.inference = std::chrono::duration<double, std::milli>(end_infer - start_infer).count();

    qDebug() << "[YoloPipeline]: backend inference done, starting post-process";
    auto start_post = std::chrono::high_resolution_clock::now();
    m_postProcessor->postProcess(out.primaryData, out.primaryShape, results, 
                                 info, m_classes, 
                                 out.secondaryData, out.secondaryShape);
    auto end_post = std::chrono::high_resolution_clock::now();
    timing.postProcess = std::chrono::duration<double, std::milli>(end_post - start_post).count();
    
    timing.total = timing.preProcess + timing.inference + timing.postProcess;
    qDebug() << "[YoloPipeline]: runInference completed. Results:" << results.size();
    return nullptr; // OK
}

void YoloPipeline::warmUp() {
    if (m_backend) {
        m_backend->warmUp(m_imgSize);
    }
}
