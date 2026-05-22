#include <gtest/gtest.h>
#include "FFmpegVideoSource.h"
#include "SourceConfig.h"

TEST(FFmpegVideoSourceTest, OpenInvalidPath) {
    FFmpegVideoSource source;
    SourceConfig config;
    config.sourceType = InputSourceType::VideoFile;
    config.filePath = "invalid_path.mp4";

    EXPECT_FALSE(source.open(config));
}

TEST(FFmpegVideoSourceTest, InitialState) {
    FFmpegVideoSource source;
    EXPECT_EQ(source.currentFrameIndex(), 0);
    EXPECT_EQ(source.frameCount(), 0);
    EXPECT_NEAR(source.nativeFps(), 0.0, 0.001);
}
