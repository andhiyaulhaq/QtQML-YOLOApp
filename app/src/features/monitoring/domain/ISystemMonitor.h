#pragma once

#include "SystemStats.h"

class ISystemMonitor {
public:
    virtual ~ISystemMonitor() = default;
    virtual void initialize() = 0;
    virtual void cleanup() = 0;
    virtual SystemStats poll() = 0;
};
