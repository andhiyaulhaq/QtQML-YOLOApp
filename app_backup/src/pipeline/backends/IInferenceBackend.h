#pragma once

#include "../YoloTypes.h"
#include <vector>

struct InferenceOutput {
    void* primaryData;
    std::vector<int64_t> primaryShape;
    void* secondaryData;            // nullptr if no second output (e.g. for segment masks)
    std::vector<int64_t> secondaryShape;
};

class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;

    virtual const char* createSession(DL_INIT_PARAM& params) = 0;
    
    virtual InferenceOutput runInference(float* blobData, const std::vector<int64_t>& inputDims) = 0;
    
    virtual void warmUp(const std::vector<int>& imgSize) = 0;
    
    virtual std::vector<int64_t> getOutputShape() const = 0;
};
