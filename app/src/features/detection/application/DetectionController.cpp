#include "DetectionController.h"
#include <QDebug>

DetectionController::DetectionController(InferenceWorker *worker, QObject *parent)
    : QObject(parent)
    , m_worker(worker)
    , m_model(new DetectionListModel(this))
{
    m_lastInferenceTime = std::chrono::steady_clock::now();
}

void DetectionController::setCurrentTask(YoloTask::TaskType task)
{
    qDebug() << "DetectionController::setCurrentTask called with task:" << (int)task;
    if (m_currentTask != task) {
        m_currentTask = task;
        emit currentTaskChanged();
        if (m_currentTask != static_cast<YoloTask::TaskType>(-1) && 
            m_currentRuntime != static_cast<YoloTask::RuntimeType>(-1)) {
            emit requestModelChange(createCurrentConfig());
        }
    }
}

void DetectionController::setCurrentRuntime(YoloTask::RuntimeType runtime)
{
    if (m_currentRuntime != runtime) {
        m_currentRuntime = runtime;
        emit currentRuntimeChanged();
        if (m_currentTask != static_cast<YoloTask::TaskType>(-1) && 
            m_currentRuntime != static_cast<YoloTask::RuntimeType>(-1)) {
            emit requestModelChange(createCurrentConfig());
        }
    }
}

void DetectionController::updateDetections(const std::vector<DetectionResult>& results, 
                                          const std::vector<std::string>& classNames, 
                                          const InferenceTiming& timing, 
                                          const QSize& frameSize)
{
    m_model->updateDetections(results, classNames, frameSize);
    
    m_preProcessTime = timing.preProcess;
    m_inferenceTime = timing.inference;
    m_postProcessTime = timing.postProcess;
    emit timingChanged();

    auto now = std::chrono::steady_clock::now();
    double diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastInferenceTime).count();
    
    if (diff > 0) {
        double currentFps = 1000.0 / diff;
        if (m_inferenceFps == 0) m_inferenceFps = currentFps;
        else m_inferenceFps = m_inferenceFps * 0.9 + currentFps * 0.1;
        emit inferenceFpsChanged();
    }
    m_lastInferenceTime = now;
}

InferenceConfig DetectionController::createCurrentConfig() const
{
    InferenceConfig config;
    config.taskType = m_currentTask;
    config.runtimeType = m_currentRuntime;
    
    std::string taskDir;
    std::string modelName;
    
    switch (m_currentTask) {
        case YoloTask::TaskType::ObjectDetection:
            taskDir = "detection";
            modelName = "yolov8n";
            break;
        case YoloTask::TaskType::PoseEstimation:
            taskDir = "pose";
            modelName = "yolov8n-pose";
            break;
        case YoloTask::TaskType::ImageSegmentation:
            taskDir = "segmentation";
            modelName = "yolov8n-seg";
            break;
    }
    
    if (m_currentRuntime == YoloTask::RuntimeType::OpenVINO) {
        config.modelPath = "assets/openvino/" + taskDir + "/" + modelName + ".xml";
    } else {
        config.modelPath = "assets/onnx/" + taskDir + "/" + modelName + ".onnx";
    }
    
    return config;
}
