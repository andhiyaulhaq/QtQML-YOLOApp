#include "InferenceWorker.h"
#include <QDebug>
#include "../../shared/domain/UiLogger.h"
#include <fstream>
#include <thread>
#include <QSize>

InferenceWorker::InferenceWorker(IDetectionModel *model, QObject *parent)
    : QObject(parent)
    , m_model(model)
{
}

InferenceWorker::~InferenceWorker()
{
    stopInference();
}

void InferenceWorker::startInference(const InferenceConfig& config)
{
    if (!m_model) return;

    m_running = false; 
    while (m_isProcessing.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto t0 = std::chrono::steady_clock::now();
    UiLogger::ctrl("InferenceWorker: Requesting session → \"" + QString::fromStdString(config.modelPath) + "\"");
    
    const char* status = m_model->createSession(config);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    if (status != nullptr) {
        UiLogger::ctrl("InferenceWorker: Session FAILED → " + QString(status));
        emit errorOccurred("Initialization Error", QString("[YoloPipeline]: %1").arg(status));
        return;
    }

    m_running = true;
    UiLogger::ctrl("InferenceWorker: Session created → OK (" + QString::number(elapsed) + " ms)");
    emit modelLoaded(config.taskType, config.runtimeType);
}

void InferenceWorker::stopInference()
{
    m_running = false;
}

void InferenceWorker::processFrame(std::shared_ptr<cv::Mat> frame)
{
    if (!m_running || !m_model || !frame) return;

    bool expected = false;
    if (!m_isProcessing.compare_exchange_strong(expected, true)) {
        return; 
    }

    std::vector<DetectionResult> results;
    InferenceTiming timing;
    
    QSize frameSize(frame->cols, frame->rows);
    m_model->runInference(*frame, results, timing);

    emit detectionsReady(results, m_model->classNames(), timing, frameSize);
    emit latestDetectionsReady(std::make_shared<std::vector<DetectionResult>>(results), frameSize);

    m_isProcessing = false;
}
