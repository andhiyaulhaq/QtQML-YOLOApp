#pragma once

#include <QObject>
#include <QQmlEngine>
#include "InferenceWorker.h"
#include "../ui/InferenceListModel.h"
#include "../domain/TaskType.h"
#include "../domain/InferenceConfig.h"
#include "../domain/InferenceResult.h"
#include <chrono>
#include <QSize>

class InferenceController : public QObject {
    Q_OBJECT

    Q_PROPERTY(QObject* visionObjects READ visionObjects NOTIFY visionObjectsChanged)
    Q_PROPERTY(YoloTask::TaskType currentTask READ currentTask WRITE setCurrentTask NOTIFY currentTaskChanged)
    Q_PROPERTY(YoloTask::RuntimeType currentRuntime READ currentRuntime WRITE setCurrentRuntime NOTIFY currentRuntimeChanged)
    Q_PROPERTY(double preProcessTime READ preProcessTime NOTIFY timingChanged)
    Q_PROPERTY(double inferenceTime READ inferenceTime NOTIFY timingChanged)
    Q_PROPERTY(double postProcessTime READ postProcessTime NOTIFY timingChanged)
    Q_PROPERTY(double inferenceFps READ inferenceFps NOTIFY inferenceFpsChanged)

public:
    explicit InferenceController(InferenceWorker *worker, QObject *parent = nullptr);

    QObject* visionObjects() const { return m_model; }
    YoloTask::TaskType currentTask() const { return m_currentTask; }
    YoloTask::RuntimeType currentRuntime() const { return m_currentRuntime; }
    
    double preProcessTime() const { return m_preProcessTime; }
    double inferenceTime() const { return m_inferenceTime; }
    double postProcessTime() const { return m_postProcessTime; }
    double inferenceFps() const { return m_inferenceFps; }

public slots:
    void setCurrentTask(YoloTask::TaskType task);
    void setCurrentRuntime(YoloTask::RuntimeType runtime);
    void updateDetections(const std::vector<InferenceResult>& results, 
                          const std::vector<std::string>& classNames, 
                          const InferenceTiming& timing, 
                          const QSize& frameSize);

signals:
    void visionObjectsChanged();
    void currentTaskChanged();
    void currentRuntimeChanged();
    void timingChanged();
    void inferenceFpsChanged();
    
    // Internal signal to trigger worker change
    void requestModelChange(const InferenceConfig& config);

private:
    InferenceWorker *m_worker;
    InferenceListModel *m_model;
    
    YoloTask::TaskType m_currentTask = static_cast<YoloTask::TaskType>(-1);
    YoloTask::RuntimeType m_currentRuntime = static_cast<YoloTask::RuntimeType>(-1);
    
    double m_preProcessTime = 0.0;
    double m_inferenceTime = 0.0;
    double m_postProcessTime = 0.0;
    double m_inferenceFps = 0.0;
    
    std::chrono::time_point<std::chrono::steady_clock> m_lastInferenceTime;
    
    InferenceConfig createCurrentConfig() const;
    void resetFps();
};
