#include <gtest/gtest.h>
#include "SimdUtils.h"
#include <vector>
#include <numeric>

TEST(SimdUtilsTest, HwcToChwConversion) {
    int width = 16;
    int height = 16;
    int step = width * 3;
    std::vector<uint8_t> src(height * step);
    
    // Fill with pattern
    for (int i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>(i % 256);
    }

    std::vector<float> dst(3 * width * height, 0.0f);
    simd::hwc_to_chw_bgr_to_rgb_sse41(src.data(), dst.data(), width, height, step);

    // Naive reference check
    const float kInv255 = 1.0f / 255.0f;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            uint8_t b = src[h * step + w * 3 + 0];
            uint8_t g = src[h * step + w * 3 + 1];
            uint8_t r = src[h * step + w * 3 + 2];

            float expected_r = r * kInv255;
            float expected_g = g * kInv255;
            float expected_b = b * kInv255;

            EXPECT_NEAR(dst[0 * width * height + h * width + w], expected_r, 1e-5);
            EXPECT_NEAR(dst[1 * width * height + h * width + w], expected_g, 1e-5);
            EXPECT_NEAR(dst[2 * width * height + h * width + w], expected_b, 1e-5);
        }
    }
}

TEST(SimdUtilsTest, UpdateBestScores) {
    int n = 100;
    std::vector<float> current_scores(n);
    std::vector<float> best_scores(n, 0.5f);
    std::vector<int> best_class_ids(n, -1);

    // Set some scores above 0.5
    for (int i = 0; i < n; ++i) {
        current_scores[i] = (i % 2 == 0) ? 0.9f : 0.1f;
    }

    simd::update_best_scores_sse41(current_scores.data(), best_scores.data(), best_class_ids.data(), 5, n);

    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            EXPECT_FLOAT_EQ(best_scores[i], 0.9f);
            EXPECT_EQ(best_class_ids[i], 5);
        } else {
            EXPECT_FLOAT_EQ(best_scores[i], 0.5f);
            EXPECT_EQ(best_class_ids[i], -1);
        }
    }
}

TEST(SimdUtilsTest, CheckThreshold) {
    float threshold = 0.5f;
    
    // Case 1: All above
    float scores1[4] = {0.6f, 0.7f, 0.8f, 0.9f};
    EXPECT_EQ(simd::check_threshold_sse41(scores1, threshold), 0xF); // 1111

    // Case 2: None above
    float scores2[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    EXPECT_EQ(simd::check_threshold_sse41(scores2, threshold), 0x0); // 0000

    // Case 3: Mixed
    float scores3[4] = {0.1f, 0.6f, 0.2f, 0.7f};
    EXPECT_EQ(simd::check_threshold_sse41(scores3, threshold), 0xA); // 1010
}
