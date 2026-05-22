#include <gtest/gtest.h>
#include "PostProcessor.h"
#include <vector>

TEST(DetectionPostProcessorTest, BasicDetection) {
    // Setup: 1 detection, 80 classes (COCO), 8400 strides (YOLOv8 default for 640x640)
    int numClasses = 80;
    int strideNum = 8400;
    int signalResultNum = 4 + numClasses; // cx, cy, w, h + scores

    std::vector<float> mockOutput(signalResultNum * strideNum, 0.0f);

    // Create a mock detection at stride 100
    int targetIdx = 100;
    // Box: cx=100, cy=100, w=50, h=50
    mockOutput[0 * strideNum + targetIdx] = 100.0f;
    mockOutput[1 * strideNum + targetIdx] = 100.0f;
    mockOutput[2 * strideNum + targetIdx] = 50.0f;
    mockOutput[3 * strideNum + targetIdx] = 50.0f;
    // Class 5 (bus) score = 0.9
    mockOutput[(4 + 5) * strideNum + targetIdx] = 0.9f;

    DetectionPostProcessor processor(YoloTask::TaskType::ObjectDetection, 0.5f, 0.45f);
    processor.initBuffers(strideNum);

    std::vector<InferenceResult> results;
    LetterboxInfo info;
    info.scale = 1.0f;
    info.padW = 0;
    info.padH = 0;

    std::vector<std::string> classes(numClasses, "class");
    std::vector<int64_t> outputDims = {1, signalResultNum, strideNum};

    processor.postProcess(mockOutput.data(), outputDims, results, info, classes);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].classId, 5);
    EXPECT_NEAR(results[0].confidence, 0.9f, 1e-5);
    EXPECT_EQ(results[0].box.x, 75); // 100 - 50/2
    EXPECT_EQ(results[0].box.y, 75); // 100 - 50/2
    EXPECT_EQ(results[0].box.width, 50);
    EXPECT_EQ(results[0].box.height, 50);
}

TEST(DetectionPostProcessorTest, NMS_Suppression) {
    int numClasses = 80;
    int strideNum = 100;
    int signalResultNum = 4 + numClasses;

    std::vector<float> mockOutput(signalResultNum * strideNum, 0.0f);

    // Detection 1: strong
    mockOutput[0 * strideNum + 10] = 100.0f;
    mockOutput[1 * strideNum + 10] = 100.0f;
    mockOutput[2 * strideNum + 10] = 50.0f;
    mockOutput[3 * strideNum + 10] = 50.0f;
    mockOutput[(4 + 0) * strideNum + 10] = 0.9f;

    // Detection 2: overlapping with 1, weaker
    mockOutput[0 * strideNum + 11] = 102.0f;
    mockOutput[1 * strideNum + 11] = 102.0f;
    mockOutput[2 * strideNum + 11] = 50.0f;
    mockOutput[3 * strideNum + 11] = 50.0f;
    mockOutput[(4 + 0) * strideNum + 11] = 0.8f;

    DetectionPostProcessor processor(YoloTask::TaskType::ObjectDetection, 0.5f, 0.45f);
    processor.initBuffers(strideNum);

    std::vector<InferenceResult> results;
    LetterboxInfo info;
    std::vector<std::string> classes(numClasses, "class");
    std::vector<int64_t> outputDims = {1, signalResultNum, strideNum};

    processor.postProcess(mockOutput.data(), outputDims, results, info, classes);

    // Only the strongest should remain after NMS
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0].confidence, 0.9f, 1e-5);
}
