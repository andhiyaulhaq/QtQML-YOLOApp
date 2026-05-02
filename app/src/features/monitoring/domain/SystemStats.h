#pragma once

#include <QString>

struct SystemStats {
    double  cpuPercent    = 0.0;
    QString systemMemory;
    QString processMemory;

    QString formatted() const {
        return QString("CPU: %1%\nSYS: %2\nAPP: %3")
            .arg(cpuPercent, 0, 'f', 1)
            .arg(systemMemory)
            .arg(processMemory);
    }
};
