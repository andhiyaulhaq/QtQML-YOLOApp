#pragma once

#include "IInferenceBackend.h"
#include <openvino/openvino.hpp>
#include <memory>
#include <vector>

class OpenVinoBackend : public IInferenceBackend {
public:
    OpenVinoBackend();
    ~OpenVinoBackend() override;

    const char* createSession(const InferenceConfig& config) override;
    InferenceOutput runInference(float* blobData, const std::vector<int64_t>& inputDims) override;
    void warmUp(const std::vector<int>& imgSize) override;
    std::vector<int64_t> getOutputShape() const override;

private:
    ov::Core m_core;
    ov::CompiledModel m_compiledModel;
    ov::InferRequest m_inferRequest;
    
    YoloTask::TaskType m_taskType;
};
