#include "OpenVinoBackend.h"
#include <iostream>

OpenVinoBackend::OpenVinoBackend() {}

OpenVinoBackend::~OpenVinoBackend() {}

const char* OpenVinoBackend::createSession(DL_INIT_PARAM& iParams) {
    try {
        m_modelType = iParams.modelType;

        std::shared_ptr<ov::Model> model = m_core.read_model(iParams.modelPath);
        
        // Performance hints
        ov::AnyMap config = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};
        
        // CPU is default. OpenVINO handles dynamic switching if "AUTO" is used, but "CPU" is safer for consistent results.
        m_compiledModel = m_core.compile_model(model, "CPU", config);
        m_inferRequest = m_compiledModel.create_infer_request();

        return RET_OK;
    } catch (const std::exception& e) {
        std::cerr << "[OpenVINO]: Create session failed: " << e.what() << std::endl;
        return "Create session failed.";
    }
}

InferenceOutput OpenVinoBackend::runInference(float* blobData, const std::vector<int64_t>& inputDims) {
    // 1. Wrap blob in OpenVINO Tensor (Zero-copy)
    ov::Shape shape = { (size_t)inputDims[0], (size_t)inputDims[1], (size_t)inputDims[2], (size_t)inputDims[3] };
    ov::Tensor input_tensor(ov::element::f32, shape, blobData);
    m_inferRequest.set_input_tensor(input_tensor);

    // 2. Run
    m_inferRequest.infer();

    // 3. Prepare result
    InferenceOutput output;
    
    // Primary output
    ov::Tensor out0 = m_inferRequest.get_output_tensor(0);
    output.primaryData = out0.data<float>();
    
    ov::Shape out_shape = out0.get_shape();
    output.primaryShape.clear();
    for (auto s : out_shape) output.primaryShape.push_back((int64_t)s);

    // Secondary output (for segmentation masks)
    try {
        if (m_compiledModel.outputs().size() > 1) {
            ov::Tensor out1 = m_inferRequest.get_output_tensor(1);
            output.secondaryData = out1.data<float>();
            
            ov::Shape sec_shape = out1.get_shape();
            output.secondaryShape.clear();
            for (auto s : sec_shape) output.secondaryShape.push_back((int64_t)s);
        } else {
            output.secondaryData = nullptr;
        }
    } catch (...) {
        output.secondaryData = nullptr;
    }

    return output;
}

void OpenVinoBackend::warmUp(const std::vector<int>& imgSize) {
    cv::Mat dummy = cv::Mat::zeros(imgSize[0], imgSize[1], CV_32FC3);
    std::vector<int64_t> dims = {1, 3, (int64_t)imgSize[0], (int64_t)imgSize[1]};
    runInference((float*)dummy.data, dims);
}

std::vector<int64_t> OpenVinoBackend::getOutputShape() const {
    if (m_compiledModel.outputs().empty()) return {};
    ov::Shape s = m_compiledModel.output(0).get_shape();
    std::vector<int64_t> result;
    for (auto val : s) result.push_back((int64_t)val);
    return result;
}
