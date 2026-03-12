#pragma once

#include "inference_backend.h"
#include "onnxruntime_cxx_api.h"
#include <atomic>
#include <memory>
#include <string>

class OnnxRuntimeBackend : public IInferenceBackend {
public:
    OnnxRuntimeBackend();
    ~OnnxRuntimeBackend() override;

    const char* createSession(DL_INIT_PARAM& params) override;
    InferenceOutput runInference(float* blobData, const std::vector<int64_t>& inputDims) override;
    void warmUp(const std::vector<int>& imgSize) override;
    std::vector<int64_t> getOutputShape() const override;

private:
    Ort::Env m_env;
    std::vector<Ort::Session*> m_sessionPool;
    std::vector<std::string> m_inputNodeNameStorage;
    std::vector<std::string> m_outputNodeNameStorage;
    std::vector<const char*> m_inputNodeNames;
    std::vector<const char*> m_outputNodeNames;
    Ort::RunOptions m_options;
    
    std::atomic<size_t> m_sessionIndex{0};
    bool m_cudaEnable = false;
    MODEL_TYPE m_modelType;
    std::vector<Ort::Value> m_lastOutputs; // To keep data alive
};
