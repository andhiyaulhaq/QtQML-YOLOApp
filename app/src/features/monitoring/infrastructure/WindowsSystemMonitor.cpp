#include "WindowsSystemMonitor.h"
#include <QDateTime>

WindowsSystemMonitor::WindowsSystemMonitor()
{
    initialize();
}

WindowsSystemMonitor::~WindowsSystemMonitor()
{
    cleanup();
}

void WindowsSystemMonitor::initialize()
{
#ifdef Q_OS_WIN
    PdhOpenQuery(nullptr, NULL, &m_cpuQuery);
    PdhAddEnglishCounter(m_cpuQuery, L"\\Processor(_Total)\\% Processor Time", NULL, &m_cpuCounter);
    PdhCollectQueryData(m_cpuQuery);
    m_self = GetCurrentProcess();
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    m_numProcessors = sysInfo.dwNumberOfProcessors;
    
    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;
    GetSystemTimeAsFileTime(&ftime);
    memcpy(&now, &ftime, sizeof(FILETIME));
    GetProcessTimes(m_self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&sys, &fsys, sizeof(FILETIME));
    memcpy(&user, &fuser, sizeof(FILETIME));
    m_lastCPU     = now;
    m_lastSysCPU  = sys;
    m_lastUserCPU = user;
#endif
}

void WindowsSystemMonitor::cleanup()
{
#ifdef Q_OS_WIN
    if (m_cpuQuery) PdhCloseQuery(m_cpuQuery);
#endif
}

SystemStats WindowsSystemMonitor::poll()
{
    SystemStats stats;
    stats.cpuPercent    = getCpuUsage();
    stats.systemMemory  = getSystemMemoryInfo();
    stats.processMemory = getProcessMemoryInfo();
    return stats;
}

double WindowsSystemMonitor::getCpuUsage()
{
#ifdef Q_OS_WIN
    PDH_FMT_COUNTERVALUE counterVal;
    PdhCollectQueryData(m_cpuQuery);
    PdhGetFormattedCounterValue(m_cpuCounter, PDH_FMT_DOUBLE, nullptr, &counterVal);
    return counterVal.doubleValue;
#else
    return 0.0;
#endif
}

QString WindowsSystemMonitor::getSystemMemoryInfo()
{
#ifdef Q_OS_WIN
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    double totalGB = memInfo.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    double usedGB = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
    return QString("%1GB/%2GB").arg(QString::number(usedGB, 'f', 1)).arg(QString::number(totalGB, 'f', 1));
#else
    return "N/A";
#endif
}

QString WindowsSystemMonitor::getProcessMemoryInfo()
{
#ifdef Q_OS_WIN
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(m_self, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        double rssMB = pmc.WorkingSetSize / (1024.0 * 1024.0);
        return QString("%1MB RSS").arg(QString::number(rssMB, 'f', 1));
    }
    return "N/A";
#else
    return "N/A";
#endif
}
