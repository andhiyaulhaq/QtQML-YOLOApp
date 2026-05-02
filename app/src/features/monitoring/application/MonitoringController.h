#pragma once

#include <QObject>
#include <QQmlEngine>
#include "../domain/SystemStats.h"

class SystemMonitorWorker;

class MonitoringController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString statsText READ statsText NOTIFY statsTextChanged)

public:
    explicit MonitoringController(SystemMonitorWorker *worker, QObject *parent = nullptr);
    ~MonitoringController() override;

    QString statsText() const { return m_statsText; }

public slots:
    void updateStats(const SystemStats& stats);

signals:
    void statsTextChanged();

private:
    SystemMonitorWorker *m_worker;
    QString m_statsText = "Initializing...";
};
