#pragma once

#include "inference_backend.h"
#include <openvino/openvino.hpp>
#include <memory>
#include <vector>

class OpenVinoBackend : public IInferenceBackend {
public:
    OpenVinoBackend();
    ~OpenVinoBackend() override;

    const char* createSession(DL_INIT_PARAM& params) override;
    InferenceOutput runInference(float* blobData, const std::vector<int64_t>& inputDims) override;
    void warmUp(const std::vector<int>& imgSize) override;
    std::vector<int64_t> getOutputShape() const override;

private:
    ov::Core m_core;
    ov::CompiledModel m_compiledModel;
    ov::InferRequest m_inferRequest;
    
    // OpenVINO output memory can be volatile; we might need to store it if we want it to persist.
    // However, given the current sequential pipeline, the request buffer should be stable.
    
    MODEL_TYPE m_modelType;
};
