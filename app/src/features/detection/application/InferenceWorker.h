#pragma once

#include <QObject>
#include <memory>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "../domain/IDetectionModel.h"
#include "../domain/DetectionResult.h"
#include "../domain/InferenceConfig.h"

class InferenceWorker : public QObject {
    Q_OBJECT

public:
    explicit InferenceWorker(IDetectionModel *model, QObject *parent = nullptr);
    ~InferenceWorker() override;

signals:
    void detectionsReady(const std::vector<DetectionResult>& results, 
                         const std::vector<std::string>& classNames, 
                         const InferenceTiming& timing, 
                         const QSize& frameSize);
    
    void latestDetectionsReady(std::shared_ptr<std::vector<DetectionResult>> results, 
                               const QSize& frameSize);
    
    void modelLoaded(YoloTask::TaskType taskType, YoloTask::RuntimeType runtimeType);
    void errorOccurred(const QString& title, const QString& message);

public slots:
    void startInference(const InferenceConfig& config);
    void stopInference();
    void processFrame(std::shared_ptr<cv::Mat> frame);
    
    std::atomic<bool>* getProcessingFlag() { return &m_isProcessing; }

private:
    IDetectionModel *m_model;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isProcessing{false};
};
