#include "SystemMonitorWorker.h"

SystemMonitorWorker::SystemMonitorWorker(ISystemMonitor *monitor, QObject *parent)
    : QObject(parent)
    , m_monitor(monitor)
    , m_timer(new QTimer(this))
{
    m_timer->setInterval(1000);
    connect(m_timer, &QTimer::timeout, this, &SystemMonitorWorker::onTimeout);
}

SystemMonitorWorker::~SystemMonitorWorker()
{
    stop();
}

void SystemMonitorWorker::start()
{
    if (!m_timer->isActive()) {
        m_timer->start();
        onTimeout(); // Initial poll
    }
}

void SystemMonitorWorker::stop()
{
    m_timer->stop();
}

void SystemMonitorWorker::onTimeout()
{
    if (m_monitor) {
        emit statsUpdated(m_monitor->poll());
    }
}
