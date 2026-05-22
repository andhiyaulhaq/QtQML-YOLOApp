#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "CaptureWorker.h"
#include "ICaptureSource.h"

using ::testing::Return;
using ::testing::_;
using ::testing::AtLeast;

class MockCaptureSource : public ICaptureSource {
public:
    MOCK_METHOD(bool, open, (const SourceConfig& config), (override));
    MOCK_METHOD(void, close, (), (override));
    MOCK_METHOD(bool, readFrame, (cv::Mat& outFrame), (override));
    MOCK_METHOD(QSize, currentResolution, (), (const, override));
    MOCK_METHOD(int64_t, frameCount, (), (const, override));
    MOCK_METHOD(int64_t, currentFrameIndex, (), (const, override));
    MOCK_METHOD(double, nativeFps, (), (const, override));
    MOCK_METHOD(bool, seekToFrame, (int64_t frameIndex), (override));
};

TEST(CaptureWorkerTest, Initialization) {
    auto mockSource = std::make_unique<MockCaptureSource>();
    EXPECT_CALL(*mockSource, close()).Times(AtLeast(0));
    
    CaptureWorker worker(mockSource.get());
    // mockSource will be deleted at end of scope, worker doesn't own it but we pass it.
    // Actually CaptureWorker takes raw pointer.
}

TEST(CaptureWorkerTest, SetSourceUpdatesConfig) {
    auto mockSource = std::make_unique<MockCaptureSource>();
    CaptureWorker worker(nullptr);

    SourceConfig config;
    config.sourceType = InputSourceType::VideoFile;
    config.filePath = "test.mp4";

    worker.setSource(mockSource.get(), config);
}
