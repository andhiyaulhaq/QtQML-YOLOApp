#include "OpenVinoBackend.h"
#include <iostream>
#include <opencv2/opencv.hpp>

OpenVinoBackend::OpenVinoBackend() {}

OpenVinoBackend::~OpenVinoBackend() {}

const char* OpenVinoBackend::createSession(const InferenceConfig& config) {
    try {
        m_taskType = config.taskType;

        std::shared_ptr<ov::Model> model = m_core.read_model(config.modelPath);
        
        ov::AnyMap ovConfig = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};
        
        m_compiledModel = m_core.compile_model(model, "CPU", ovConfig);
        m_inferRequest = m_compiledModel.create_infer_request();

        return nullptr; // OK
    } catch (const std::exception& e) {
        std::cerr << "[OpenVINO]: Create session failed: " << e.what() << std::endl;
        return "Create session failed.";
    }
}

InferenceOutput OpenVinoBackend::runInference(float* blobData, const std::vector<int64_t>& inputDims) {
    ov::Shape shape = { (size_t)inputDims[0], (size_t)inputDims[1], (size_t)inputDims[2], (size_t)inputDims[3] };
    ov::Tensor input_tensor(ov::element::f32, shape, blobData);
    m_inferRequest.set_input_tensor(input_tensor);

    m_inferRequest.infer();

    InferenceOutput output;
    
    ov::Tensor out0 = m_inferRequest.get_output_tensor(0);
    output.primaryData = out0.data<float>();
    
    ov::Shape out_shape = out0.get_shape();
    output.primaryShape.clear();
    for (auto s : out_shape) output.primaryShape.push_back((int64_t)s);

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
