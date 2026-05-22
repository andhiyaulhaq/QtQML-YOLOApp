#include <gtest/gtest.h>
#include "PreProcessor.h"
#include <opencv2/opencv.hpp>

TEST(PreProcessorTest, LetterboxCalculation) {
    // 640x640 target
    ImagePreProcessor processor(YoloTask::TaskType::ObjectDetection, {640, 640});

    // Input 1280x720 (16:9)
    cv::Mat input(720, 1280, CV_8UC3);
    cv::Mat output;
    
    LetterboxInfo info = processor.preProcess(input, output);

    // Scale should be 1280 / 640 = 2.0
    EXPECT_NEAR(info.scale, 2.0f, 1e-5);
    
    // Resized image should be 640x360
    // Padding H: (640 - 360) / 2 = 140
    // Padding W: (640 - 640) / 2 = 0
    EXPECT_EQ(info.padW, 0);
    EXPECT_EQ(info.padH, 140);
    
    EXPECT_EQ(output.cols, 640);
    EXPECT_EQ(output.rows, 640);
}

TEST(PreProcessorTest, BlobConversion) {
    ImagePreProcessor processor(YoloTask::TaskType::ObjectDetection, {640, 640});
    
    // 4x4 image for simple testing
    cv::Mat img(4, 4, CV_8UC3, cv::Scalar(0, 128, 255)); // B=0, G=128, R=255
    std::vector<float> blob(3 * 4 * 4);
    
    processor.preProcessImageToBlob(img, blob.data());
    
    // Check first pixel in each channel (RGB order)
    // R: 255/255 = 1.0
    // G: 128/255 = 0.50196...
    // B: 0/255 = 0.0
    EXPECT_NEAR(blob[0], 1.0f, 1e-5);           // R channel
    EXPECT_NEAR(blob[4*4], 0.50196f, 1e-4);    // G channel
    EXPECT_NEAR(blob[2*4*4], 0.0f, 1e-5);      // B channel
}
