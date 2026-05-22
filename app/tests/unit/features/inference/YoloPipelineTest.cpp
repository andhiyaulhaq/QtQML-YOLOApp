#include <gtest/gtest.h>
#include "YoloPipeline.h"
#include "InferenceConfig.h"

TEST(YoloPipelineTest, ChineseCharactersInPath) {
    YoloPipeline pipeline;
    InferenceConfig config;
    config.modelPath = "模型/yolov8n.onnx"; // Chinese path

    const char* status = pipeline.createSession(config);
    ASSERT_NE(status, nullptr);
    EXPECT_STREQ(status, "[YoloPipeline]: Model path cannot contain Chinese characters.");
}

TEST(YoloPipelineTest, UnsupportedTaskType) {
    YoloPipeline pipeline;
    InferenceConfig config;
    config.taskType = static_cast<YoloTask::TaskType>(999); // Invalid

    // This should probably throw or return error
    // In implementation it throws runtime_error and returns "[YoloPipeline]: Create session failed."
    const char* status = pipeline.createSession(config);
    ASSERT_NE(status, nullptr);
}
