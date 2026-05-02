#pragma once

#include <QObject>
#include <QTimer>
#include <QDateTime>
#include <QDebug>
#include "SystemStats.h"

#ifdef Q_OS_WIN
#include <windows.h>
#include <pdh.h>
#include <psapi.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")
#endif

class SystemMonitorInfra : public QObject {
    Q_OBJECT

public:
    explicit SystemMonitorInfra(QObject *parent = nullptr);
    ~SystemMonitorInfra();

public slots:
    void startMonitoring();
    void stopMonitoring();

signals:
    void statsUpdated(SystemStats stats);

private slots:
    void updateMetrics();

private:
    QTimer *m_timer;

#ifdef Q_OS_WIN
    PDH_HQUERY   m_cpuQuery;
    PDH_HCOUNTER m_cpuCounter;
    ULARGE_INTEGER m_lastCPU, m_lastSysCPU, m_lastUserCPU;
    int    m_numProcessors;
    HANDLE m_self;
#endif

    double  getCpuUsage();
    QString getSystemMemoryInfo();
    QString getProcessMemoryInfo();

    void initializePlatform();
    void cleanupPlatform();
};
