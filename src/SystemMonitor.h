#ifndef SYSTEMMONITOR_H
#define SYSTEMMONITOR_H

#include <QObject>
#include <QTimer>
#include <QDateTime>
#include <QDebug>

#ifdef Q_OS_WIN
#include <windows.h>
#include <pdh.h>
#include <psapi.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")
#elif defined(Q_OS_LINUX)
#include <sys/sysinfo.h>
#include <fstream>
#include <unistd.h>
#elif defined(Q_OS_MACOS)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>
#endif

class SystemMonitor : public QObject {
    Q_OBJECT

public:
    explicit SystemMonitor(QObject *parent = nullptr);
    ~SystemMonitor();

public slots:
    void startMonitoring();
    void stopMonitoring();

signals:
    void resourceUsageUpdated(QString formattedStats);

private slots:
    void updateMetrics();

private:
    QTimer *m_timer;
    
    // Platform-specific data members
#ifdef Q_OS_WIN
    PDH_HQUERY m_cpuQuery;
    PDH_HCOUNTER m_cpuCounter;
    ULARGE_INTEGER m_lastCPU, m_lastSysCPU, m_lastUserCPU;
    int m_numProcessors;
    HANDLE m_self;
#elif defined(Q_OS_LINUX)
    unsigned long long m_lastTotalUser, m_lastTotalUserLow, m_lastTotalSys, m_lastTotalIdle;
#endif
    
    // Platform-specific methods
    double getCpuUsage();
    QString getSystemMemoryInfo();
    QString getProcessMemoryInfo();
    
    void initializePlatform();
    void cleanupPlatform();
};

#endif // SYSTEMMONITOR_H