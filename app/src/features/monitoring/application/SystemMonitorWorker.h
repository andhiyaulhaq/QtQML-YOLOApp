#pragma once

#include <QObject>
#include <QTimer>
#include "../domain/ISystemMonitor.h"

class SystemMonitorWorker : public QObject {
    Q_OBJECT

public:
    explicit SystemMonitorWorker(ISystemMonitor *monitor, QObject *parent = nullptr);
    ~SystemMonitorWorker() override;

public slots:
    void start();
    void stop();

signals:
    void statsUpdated(const SystemStats& stats);

private slots:
    void onTimeout();

private:
    ISystemMonitor *m_monitor;
    QTimer *m_timer;
};
