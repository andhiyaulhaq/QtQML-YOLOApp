#include <gtest/gtest.h>
#include "OpenCVImageFileSource.h"
#include "SourceConfig.h"

TEST(OpenCVImageFileSourceTest, OpenNonExistentFile) {
    OpenCVImageFileSource source;
    SourceConfig config;
    config.sourceType = InputSourceType::ImageFile;
    config.filePath = "non_existent.jpg";

    EXPECT_FALSE(source.open(config));
}

TEST(OpenCVImageFileSourceTest, ReadBeforeOpen) {
    OpenCVImageFileSource source;
    cv::Mat frame;
    EXPECT_FALSE(source.readFrame(frame));
}
