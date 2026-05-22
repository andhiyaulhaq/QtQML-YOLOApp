#include <gtest/gtest.h>
#include "RenderTransform.h"
#include <QSize>

// We only test the static math logic
TEST(InferenceOverlayMathTest, PreserveAspectFit_WideVideo_NarrowItem) {
    // Video: 16:9 (1.77), Item: 4:3 (1.33)
    // Result: Video height should be letterboxed (offsetY > 0)
    float itemW = 800;
    float itemH = 600;
    QSize frameSize(1920, 1080);

    RenderTransform t = RenderTransform::calculate(itemW, itemH, frameSize);

    EXPECT_FLOAT_EQ(t.renderW, 800.0f);
    EXPECT_NEAR(t.renderH, 450.0f, 0.1f); // 800 / (1920/1080) = 450
    EXPECT_FLOAT_EQ(t.offsetX, 0.0f);
    EXPECT_NEAR(t.offsetY, 75.0f, 0.1f); // (600 - 450) / 2 = 75
}

TEST(InferenceOverlayMathTest, PreserveAspectFit_NarrowVideo_WideItem) {
    // Video: 4:3 (1.33), Item: 16:9 (1.77)
    // Result: Video width should be pillarboxed (offsetX > 0)
    float itemW = 1600;
    float itemH = 900;
    QSize frameSize(800, 600);

    RenderTransform t = RenderTransform::calculate(itemW, itemH, frameSize);

    EXPECT_NEAR(t.renderW, 1200.0f, 0.1f); // 900 * (800/600) = 1200
    EXPECT_FLOAT_EQ(t.renderH, 900.0f);
    EXPECT_NEAR(t.offsetX, 200.0f, 0.1f); // (1600 - 1200) / 2 = 200
    EXPECT_FLOAT_EQ(t.offsetY, 0.0f);
}

TEST(InferenceOverlayMathTest, Stretch_WhenFrameEmpty) {
    float itemW = 800;
    float itemH = 600;
    QSize frameSize(0, 0);

    RenderTransform t = RenderTransform::calculate(itemW, itemH, frameSize);

    EXPECT_FLOAT_EQ(t.renderW, 800.0f);
    EXPECT_FLOAT_EQ(t.renderH, 600.0f);
    EXPECT_FLOAT_EQ(t.offsetX, 0.0f);
    EXPECT_FLOAT_EQ(t.offsetY, 0.0f);
}
