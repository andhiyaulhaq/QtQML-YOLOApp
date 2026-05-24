#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "VideoFileController.h"

using ::testing::_;

TEST(VideoFileControllerTest, Initialization) {
    VideoFileController controller;
    
    EXPECT_FALSE(controller.hasFile());
    EXPECT_TRUE(controller.filePath().isEmpty());
    EXPECT_EQ(controller.durationSeconds(), 0.0);
    EXPECT_EQ(controller.totalFrames(), 0);
    EXPECT_EQ(controller.currentFrame(), 0);
    EXPECT_FALSE(controller.isPaused()); // Default state
}

TEST(VideoFileControllerTest, SetFilePathStartsPaused) {
    VideoFileController controller;
    
    bool isPausedEmitted = false;
    QObject::connect(&controller, &VideoFileController::isPausedChanged, [&]() {
        isPausedEmitted = true;
    });

    controller.setFilePath(QUrl("file:///test/video.mp4"));
    
    EXPECT_TRUE(controller.hasFile());
    EXPECT_EQ(controller.filePath(), "/test/video.mp4");
    EXPECT_TRUE(controller.isPaused());
    EXPECT_TRUE(isPausedEmitted);
}

TEST(VideoFileControllerTest, TogglePlayPause) {
    VideoFileController controller;
    
    // Initialize to paused
    controller.setFilePath(QUrl("file:///test/video.mp4"));
    EXPECT_TRUE(controller.isPaused());
    
    bool requestPlayPauseEmitted = false;
    bool requestedPauseState = true;
    
    QObject::connect(&controller, &VideoFileController::requestPlayPause, [&](bool paused) {
        requestPlayPauseEmitted = true;
        requestedPauseState = paused;
    });

    // Toggle from paused to playing
    controller.togglePlayPause();
    
    EXPECT_TRUE(requestPlayPauseEmitted);
    EXPECT_FALSE(requestedPauseState); // It should request playing (paused = false)
}

TEST(VideoFileControllerTest, OnPlayStateChangedUpdatesProperty) {
    VideoFileController controller;
    
    bool isPausedEmitted = false;
    QObject::connect(&controller, &VideoFileController::isPausedChanged, [&]() {
        isPausedEmitted = true;
    });

    // Initially playing
    EXPECT_FALSE(controller.isPaused());
    
    // Worker reports paused
    controller.onPlayStateChanged(false); // false means NOT playing -> paused
    
    EXPECT_TRUE(controller.isPaused());
    EXPECT_TRUE(isPausedEmitted);
    
    isPausedEmitted = false;
    
    // Worker reports playing
    controller.onPlayStateChanged(true); // true means playing -> NOT paused
    
    EXPECT_FALSE(controller.isPaused());
    EXPECT_TRUE(isPausedEmitted);
}
