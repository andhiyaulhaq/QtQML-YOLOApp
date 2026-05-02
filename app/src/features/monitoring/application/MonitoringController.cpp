#include "MonitoringController.h"
#include <QString>

MonitoringController::MonitoringController(SystemMonitorWorker *worker, QObject *parent)
    : QObject(parent)
    , m_worker(worker)
{
}

MonitoringController::~MonitoringController()
{
}

void MonitoringController::updateStats(const SystemStats& stats)
{
    if (m_statsText != stats.formatted()) {
        m_statsText = stats.formatted();
        emit statsTextChanged();
    }
}
