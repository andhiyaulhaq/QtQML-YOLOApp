#pragma once

#include "ISystemMonitor.h"

#ifdef Q_OS_WIN
#include <windows.h>
#include <pdh.h>
#include <psapi.h>
#endif

class WindowsSystemMonitor : public ISystemMonitor {
public:
    WindowsSystemMonitor();
    ~WindowsSystemMonitor() override;

    void initialize() override;
    void cleanup() override;
    SystemStats poll() override;

private:
    double  getCpuUsage();
    QString getSystemMemoryInfo();
    QString getProcessMemoryInfo();

#ifdef Q_OS_WIN
    PDH_HQUERY   m_cpuQuery;
    PDH_HCOUNTER m_cpuCounter;
    ULARGE_INTEGER m_lastCPU, m_lastSysCPU, m_lastUserCPU;
    int    m_numProcessors;
    HANDLE m_self;
#endif
};
