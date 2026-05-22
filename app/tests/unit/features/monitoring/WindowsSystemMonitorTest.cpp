#include <gtest/gtest.h>
#include "WindowsSystemMonitor.h"

TEST(WindowsSystemMonitorTest, BasicPoll) {
    WindowsSystemMonitor monitor;
    monitor.initialize();
    
    SystemStats stats = monitor.poll();
    
    // On many systems, the first CPU poll might be 0.0 or 100.0 depending on implementation
    // But we expect memory strings to be populated
    EXPECT_FALSE(stats.systemMemory.isEmpty());
    EXPECT_FALSE(stats.processMemory.isEmpty());
    
    monitor.cleanup();
}

TEST(WindowsSystemMonitorTest, Lifecycle) {
    WindowsSystemMonitor monitor;
    // Should handle cleanup even if not initialized
    monitor.cleanup();
    SUCCEED();
}
