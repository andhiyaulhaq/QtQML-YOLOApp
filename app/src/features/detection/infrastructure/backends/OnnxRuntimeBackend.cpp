#include "OnnxRuntimeBackend.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif

OnnxRuntimeBackend::OnnxRuntimeBackend() : m_options(nullptr) {}

OnnxRuntimeBackend::~OnnxRuntimeBackend() {
    for (auto s : m_sessionPool) {
        if (s) delete s;
    }
}

const char* OnnxRuntimeBackend::createSession(const InferenceConfig& config) {
    try {
        m_taskType = config.taskType;
        m_cudaEnable = config.cudaEnable;

        m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        
        if (m_cudaEnable) {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }

        sessionOption.SetIntraOpNumThreads(config.intraOpThreads);
        sessionOption.SetInterOpNumThreads(config.interOpThreads);
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetLogSeverityLevel(3); // Default to WARNING
        sessionOption.SetExecutionMode(ORT_SEQUENTIAL);

#ifdef _WIN32
        std::string ompThreads = std::to_string(config.intraOpThreads);
        SetEnvironmentVariableA("OMP_NUM_THREADS", ompThreads.c_str());
        SetEnvironmentVariableA("KMP_AFFINITY", "granularity=fine,verbose,compact,1,0");
        SetEnvironmentVariableA("KMP_BLOCKTIME", "1");
        SetEnvironmentVariableA("KMP_SETTINGS", "1");

        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, config.modelPath.c_str(), static_cast<int>(config.modelPath.length()), nullptr, 0);
        std::wstring wide_model_path(ModelPathSize, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, config.modelPath.c_str(), static_cast<int>(config.modelPath.length()), &wide_model_path[0], ModelPathSize);
        const wchar_t* modelPath = wide_model_path.c_str();
#else
        const char* modelPath = config.modelPath.c_str();
#endif

        try {
            Ort::Session* sess = new Ort::Session(m_env, modelPath, sessionOption);
            m_sessionPool.push_back(sess);
        } catch (const std::exception& e) {
            if (m_cudaEnable) {
                std::cout << "[ONNX]: CUDA init failed (" << e.what() << "). Fallback to CPU." << std::endl;
                m_cudaEnable = false;
                sessionOption = Ort::SessionOptions();
                sessionOption.SetIntraOpNumThreads(config.intraOpThreads);
                sessionOption.SetInterOpNumThreads(config.interOpThreads);
                Ort::Session* sess = new Ort::Session(m_env, modelPath, sessionOption);
                m_sessionPool.push_back(sess);
            } else throw;
        }

        // Single session for now, pooling can be re-added if needed
        // For simplicity in porting, I'll stick to 1 session for now or use config

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::Session* primary = m_sessionPool.front();
        
        size_t inputCount = primary->GetInputCount();
        m_inputNodeNameStorage.reserve(inputCount);
        m_inputNodeNames.reserve(inputCount);
        for (size_t i = 0; i < inputCount; i++) {
            m_inputNodeNameStorage.push_back(primary->GetInputNameAllocated(i, allocator).get());
            m_inputNodeNames.push_back(m_inputNodeNameStorage.back().c_str());
        }

        size_t outputCount = primary->GetOutputCount();
        m_outputNodeNameStorage.reserve(outputCount);
        m_outputNodeNames.reserve(outputCount);
        for (size_t i = 0; i < outputCount; i++) {
            m_outputNodeNameStorage.push_back(primary->GetOutputNameAllocated(i, allocator).get());
            m_outputNodeNames.push_back(m_outputNodeNameStorage.back().c_str());
        }

        return nullptr; // OK
    } catch (const std::exception& e) {
        std::cerr << "[ONNX]: Create session failed: " << e.what() << std::endl;
        return "Create session failed.";
    }
}

InferenceOutput OnnxRuntimeBackend::runInference(float* blobData, const std::vector<int64_t>& inputDims) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), 
        blobData, 3 * inputDims[2] * inputDims[3], 
        inputDims.data(), inputDims.size());

    size_t poolSize = m_sessionPool.size();
    Ort::Session* sess = m_sessionPool[m_sessionIndex.fetch_add(1) % poolSize];
    
    m_lastOutputs = sess->Run(m_options, m_inputNodeNames.data(), &inputTensor, 1, m_outputNodeNames.data(), m_outputNodeNames.size());

    InferenceOutput output;
    
    if (m_lastOutputs.size() > 1) {
        auto shape0 = m_lastOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto shape1 = m_lastOutputs[1].GetTensorTypeAndShapeInfo().GetShape();
        
        if (shape0.size() == 4 && shape1.size() == 3) {
            std::swap(m_lastOutputs[0], m_lastOutputs[1]);
            std::swap(shape0, shape1);
        }
        
        output.primaryData = m_lastOutputs[0].GetTensorMutableData<void>();
        output.primaryShape = shape0;
        output.secondaryData = m_lastOutputs[1].GetTensorMutableData<void>();
        output.secondaryShape = shape1;
    } else {
        output.primaryData = m_lastOutputs[0].GetTensorMutableData<void>();
        output.primaryShape = m_lastOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
        output.secondaryData = nullptr;
    }

    return output; 
}

void OnnxRuntimeBackend::warmUp(const std::vector<int>& imgSize) {
    if (m_sessionPool.empty()) return;
    cv::Mat dummy = cv::Mat::zeros(imgSize[0], imgSize[1], CV_32FC3);
    float* data = (float*)dummy.data;
    std::vector<int64_t> dims = {1, 3, (int64_t)imgSize[0], (int64_t)imgSize[1]};
    runInference(data, dims);
}

std::vector<int64_t> OnnxRuntimeBackend::getOutputShape() const {
    if (m_sessionPool.empty()) return {};
    return m_sessionPool.front()->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}
