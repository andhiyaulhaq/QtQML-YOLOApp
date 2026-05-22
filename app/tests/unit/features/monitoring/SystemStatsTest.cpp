#include <gtest/gtest.h>
#include "SystemStats.h"

TEST(SystemStatsTest, FormattedOutput) {
    SystemStats stats;
    stats.cpuPercent = 45.5;
    stats.systemMemory = "12.5/16.0 GB";
    stats.processMemory = "256 MB";

    QString expected = "CPU: 45.5%\nSYS: 12.5/16.0 GB\nAPP: 256 MB";
    EXPECT_EQ(stats.formatted(), expected);
}

TEST(SystemStatsTest, FormattedOutputPrecision) {
    SystemStats stats;
    stats.cpuPercent = 45.567; // Should be rounded to 1 decimal place
    stats.systemMemory = "10 GB";
    stats.processMemory = "100 MB";

    QString expected = "CPU: 45.6%\nSYS: 10 GB\nAPP: 100 MB";
    EXPECT_EQ(stats.formatted(), expected);
}