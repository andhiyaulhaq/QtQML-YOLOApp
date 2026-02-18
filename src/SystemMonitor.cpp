#include "SystemMonitor.h"
#include <QCoreApplication>

SystemMonitor::SystemMonitor(QObject *parent)
    : QObject(parent)
    , m_timer(new QTimer(this))
{
    m_timer->setInterval(1000); // 1 second interval
    connect(m_timer, &QTimer::timeout, this, &SystemMonitor::updateMetrics);
    
    initializePlatform();
}

SystemMonitor::~SystemMonitor()
{
    cleanupPlatform();
}

void SystemMonitor::startMonitoring()
{
    if (!m_timer->isActive()) {
        // qDebug() << "[SystemMonitor] Starting resource monitoring";
        m_timer->start();
        updateMetrics(); // Initial reading
    }
}

void SystemMonitor::stopMonitoring()
{
    if (m_timer->isActive()) {
        m_timer->stop();
        // qDebug() << "[SystemMonitor] Stopped resource monitoring";
    }
}

void SystemMonitor::updateMetrics()
{
    double cpuUsage = getCpuUsage();
    QString sysMemory = getSystemMemoryInfo();
    QString processMemory = getProcessMemoryInfo();
    
    QString cpuStr = QString::number(cpuUsage, 'f', 1) + "%";
    QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    
    QString message = QString("[%1] CPU: %2 | Sys Memory: %3 | Process Memory: %4")
                     .arg(timestamp, cpuStr, sysMemory, processMemory);
    
    // qDebug() << message;
    emit resourceUsageUpdated(cpuStr, sysMemory, processMemory);
}

// Platform-specific implementations

void SystemMonitor::initializePlatform()
{
#ifdef Q_OS_WIN
    // Initialize PDH for CPU monitoring
    PdhOpenQuery(nullptr, NULL, &m_cpuQuery);
    PdhAddEnglishCounter(m_cpuQuery, L"\\Processor(_Total)\\% Processor Time", NULL, &m_cpuCounter);
    PdhCollectQueryData(m_cpuQuery);
    
    // Initialize process handle for memory monitoring
    m_self = GetCurrentProcess();
    
    // Get processor count
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    m_numProcessors = sysInfo.dwNumberOfProcessors;
    
    // Initialize CPU usage calculation variables
    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;
    
    GetSystemTimeAsFileTime(&ftime);
    memcpy(&now, &ftime, sizeof(FILETIME));
    
    GetProcessTimes(m_self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&sys, &fsys, sizeof(FILETIME));
    memcpy(&user, &fuser, sizeof(FILETIME));
    
    m_lastCPU = now;
    m_lastSysCPU = sys;
    m_lastUserCPU = user;

#elif defined(Q_OS_LINUX)
    // Initialize CPU usage calculation variables for Linux
    std::ifstream file("/proc/stat");
    if (file.is_open()) {
        std::string line;
        if (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string cpu;
            ss >> cpu >> m_lastTotalUser >> m_lastTotalUserLow >> m_lastTotalSys >> m_lastTotalIdle;
        }
        file.close();
    }
#endif
}

void SystemMonitor::cleanupPlatform()
{
#ifdef Q_OS_WIN
    if (m_cpuQuery) {
        PdhCloseQuery(m_cpuQuery);
    }
#endif
}

double SystemMonitor::getCpuUsage()
{
#ifdef Q_OS_WIN
    PDH_FMT_COUNTERVALUE counterVal;
    PdhCollectQueryData(m_cpuQuery);
    PdhGetFormattedCounterValue(m_cpuCounter, PDH_FMT_DOUBLE, nullptr, &counterVal);
    return counterVal.doubleValue;

#elif defined(Q_OS_LINUX)
    std::ifstream file("/proc/stat");
    if (!file.is_open()) return -1.0;
    
    std::string line;
    if (!std::getline(file, line)) {
        file.close();
        return -1.0;
    }
    
    std::istringstream ss(line);
    std::string cpu;
    unsigned long long totalUser, totalUserLow, totalSys, totalIdle;
    
    ss >> cpu >> totalUser >> totalUserLow >> totalSys >> totalIdle;
    file.close();
    
    unsigned long long total = (totalUser - m_lastTotalUser) + 
                             (totalUserLow - m_lastTotalUserLow) + 
                             (totalSys - m_lastTotalSys);
    unsigned long long totalAll = total + (totalIdle - m_lastTotalIdle);
    
    double percent = totalAll > 0 ? (total * 100.0 / totalAll) : 0.0;
    
    m_lastTotalUser = totalUser;
    m_lastTotalUserLow = totalUserLow;
    m_lastTotalSys = totalSys;
    m_lastTotalIdle = totalIdle;
    
    return percent;

#elif defined(Q_OS_MACOS)
    // macOS CPU monitoring using mach kernel APIs
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, 
                        (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
        unsigned long long totalTicks = cpuinfo.cpu_ticks[CPU_STATE_USER] +
                                       cpuinfo.cpu_ticks[CPU_STATE_SYSTEM] +
                                       cpuinfo.cpu_ticks[CPU_STATE_IDLE] +
                                       cpuinfo.cpu_ticks[CPU_STATE_NICE];
        
        static unsigned long long lastTotalTicks = 0;
        static unsigned long long lastIdleTicks = 0;
        
        unsigned long long totalTicksSinceLastTime = totalTicks - lastTotalTicks;
        unsigned long long idleTicksSinceLastTime = cpuinfo.cpu_ticks[CPU_STATE_IDLE] - lastIdleTicks;
        
        double cpuUsage = totalTicksSinceLastTime > 0 ? 
                         100.0 * (1.0 - (double)idleTicksSinceLastTime / totalTicksSinceLastTime) : 0.0;
        
        lastTotalTicks = totalTicks;
        lastIdleTicks = cpuinfo.cpu_ticks[CPU_STATE_IDLE];
        
        return cpuUsage;
    }
    return -1.0;

#else
    return -1.0; // Platform not supported
#endif
}

QString SystemMonitor::getSystemMemoryInfo()
{
#ifdef Q_OS_WIN
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    
    double totalGB = memInfo.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    double usedGB = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
    double usagePercent = (double)memInfo.dwMemoryLoad;
    
    return QString("%1GB/%2GB (%3%)")
           .arg(QString::number(usedGB, 'f', 1))
           .arg(QString::number(totalGB, 'f', 1))
           .arg(QString::number(usagePercent, 'f', 1));

#elif defined(Q_OS_LINUX)
    std::ifstream file("/proc/meminfo");
    if (!file.is_open()) return "N/A";
    
    std::string line;
    unsigned long long memTotal = 0, memAvailable = 0;
    
    while (std::getline(file, line)) {
        if (line.substr(0, 9) == "MemTotal:") {
            std::istringstream ss(line);
            std::string label;
            ss >> label >> memTotal;
        } else if (line.substr(0, 13) == "MemAvailable:") {
            std::istringstream ss(line);
            std::string label;
            ss >> label >> memAvailable;
            break;
        }
    }
    file.close();
    
    if (memTotal > 0) {
        unsigned long long memUsed = memTotal - memAvailable;
        double totalGB = memTotal / (1024.0 * 1024.0);
        double usedGB = memUsed / (1024.0 * 1024.0);
        double usagePercent = (double)memUsed * 100.0 / memTotal;
        
        return QString("%1GB/%2GB (%3%)")
               .arg(QString::number(usedGB, 'f', 1))
               .arg(QString::number(totalGB, 'f', 1))
               .arg(QString::number(usagePercent, 'f', 1));
    }
    return "N/A";

#elif defined(Q_OS_MACOS)
    // macOS memory monitoring
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t totalMemory = 0;
    size_t length = sizeof(totalMemory);
    
    if (sysctl(mib, 2, &totalMemory, &length, nullptr, 0) == 0) {
        vm_statistics64_data_t vmStats;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, 
                              (host_info64_t)&vmStats, &count) == KERN_SUCCESS) {
            uint64_t freeMemory = vmStats.free_count * vm_page_size;
            uint64_t usedMemory = totalMemory - freeMemory;
            
            double totalGB = totalMemory / (1024.0 * 1024.0 * 1024.0);
            double usedGB = usedMemory / (1024.0 * 1024.0 * 1024.0);
            double usagePercent = (double)usedMemory * 100.0 / totalMemory;
            
            return QString("%1GB/%2GB (%3%)")
                   .arg(QString::number(usedGB, 'f', 1))
                   .arg(QString::number(totalGB, 'f', 1))
                   .arg(QString::number(usagePercent, 'f', 1));
        }
    }
    return "N/A";

#else
    return "N/A"; // Platform not supported
#endif
}

QString SystemMonitor::getProcessMemoryInfo()
{
#ifdef Q_OS_WIN
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(m_self, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        double rssMB = pmc.WorkingSetSize / (1024.0 * 1024.0);
        double virtualMB = pmc.PrivateUsage / (1024.0 * 1024.0);
        
        return QString("%1MB RSS / %2MB Virtual")
               .arg(QString::number(rssMB, 'f', 1))
               .arg(QString::number(virtualMB, 'f', 1));
    }
    return "N/A";

#elif defined(Q_OS_LINUX)
    std::ifstream file("/proc/self/status");
    if (!file.is_open()) return "N/A";
    
    std::string line;
    unsigned long vmSize = 0, vmRSS = 0;
    
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmSize:") {
            std::istringstream ss(line);
            std::string label;
            ss >> label >> vmSize;
        } else if (line.substr(0, 5) == "VmRSS:") {
            std::istringstream ss(line);
            std::string label;
            ss >> label >> vmRSS;
        }
    }
    file.close();
    
    if (vmRSS > 0) {
        double rssMB = vmRSS / 1024.0;
        double virtualMB = vmSize / 1024.0;
        
        return QString("%1MB RSS / %2MB Virtual")
               .arg(QString::number(rssMB, 'f', 1))
               .arg(QString::number(virtualMB, 'f', 1));
    }
    return "N/A";

#elif defined(Q_OS_MACOS)
    // macOS process memory monitoring
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, 
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        double rssMB = info.resident_size / (1024.0 * 1024.0);
        double virtualMB = info.virtual_size / (1024.0 * 1024.0);
        
        return QString("%1MB RSS / %2MB Virtual")
               .arg(QString::number(rssMB, 'f', 1))
               .arg(QString::number(virtualMB, 'f', 1));
    }
    return "N/A";

#else
    return "N/A"; // Platform not supported
#endif
}