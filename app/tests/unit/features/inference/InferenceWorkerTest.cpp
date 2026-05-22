#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "InferenceWorker.h"
#include "IInferenceModel.h"
#include <thread>
#include <atomic>

using ::testing::_;
using ::testing::Return;
using ::testing::AtLeast;
using ::testing::NiceMock;

using ::testing::ReturnRef;

class MockInferenceModel : public IInferenceModel {
public:
    MOCK_METHOD(const char*, createSession, (const InferenceConfig& config), (override));
    MOCK_METHOD(char*, runInference, (const cv::Mat& frame, std::vector<InferenceResult>& results, InferenceTiming& timing), (override));
    MOCK_METHOD(const std::vector<std::string>&, classNames, (), (const, override));
    MOCK_METHOD(void, warmUp, (), (override));
};

static std::vector<std::string> g_mockClasses = {"person"};

TEST(InferenceWorkerTest, FrameDropping) {
    NiceMock<MockInferenceModel> mockModel;
    InferenceWorker worker(&mockModel);
    
    ON_CALL(mockModel, classNames()).WillByDefault(ReturnRef(g_mockClasses));

    InferenceConfig config;
    ON_CALL(mockModel, createSession(_)).WillByDefault(Return(nullptr));
    worker.startInference(config);

    std::atomic<bool> firstCallStarted{false};
    std::atomic<bool> canFinishFirstCall{false};

    EXPECT_CALL(mockModel, runInference(_, _, _))
        .WillOnce([&](const cv::Mat&, std::vector<InferenceResult>&, InferenceTiming&){
            firstCallStarted = true;
            while(!canFinishFirstCall) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return nullptr;
        });

    auto frame = std::make_shared<cv::Mat>(640, 480, CV_8UC3);
    
    // Start first inference in a thread
    std::thread t([&](){
        worker.processFrame(frame);
    });

    // Wait for first to actually start and set m_isProcessing
    while(!firstCallStarted) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Now m_isProcessing should be true. Try second call.
    // This should be dropped immediately.
    EXPECT_CALL(mockModel, runInference(_, _, _)).Times(0);
    worker.processFrame(frame);

    // Let the first one finish
    canFinishFirstCall = true;
    t.join();
}

TEST(InferenceWorkerTest, StopInferencePreventsProcessing) {
    NiceMock<MockInferenceModel> mockModel;
    InferenceWorker worker(&mockModel);
    
    worker.stopInference(); // Ensure it's stopped

    EXPECT_CALL(mockModel, runInference(_, _, _)).Times(0);
    
    auto frame = std::make_shared<cv::Mat>(640, 480, CV_8UC3);
    worker.processFrame(frame);
}
