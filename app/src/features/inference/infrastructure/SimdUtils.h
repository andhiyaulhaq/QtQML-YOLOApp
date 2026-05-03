#pragma once

#include <immintrin.h>
#include <cstdint>

/**
 * @brief SIMD Utility functions for YOLO preprocessing and postprocessing.
 */
namespace simd {

/**
 * @brief Fused loop for HWC to CHW conversion and normalization [0, 1].
 * Optimized with SSE4.1 (16 pixels per iteration).
 */
inline void hwc_to_chw_bgr_to_rgb_sse41(const uint8_t* src, float* dst, int width, int height, int step) {
    const float kInv255 = 1.0f / 255.0f;
    const int plane_size = width * height;
    float* dst_r = dst;
    float* dst_g = dst + plane_size;
    float* dst_b = dst + 2 * plane_size;

    __m128 v_inv255 = _mm_set1_ps(kInv255);

    __m128i mask_r = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i mask_g = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i mask_b = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    for (int h = 0; h < height; ++h) {
        const uint8_t* row_ptr = src + h * step;
        int w = 0;
        
        for (; w <= width - 4; w += 4) {
            const uint8_t* p = row_ptr + w * 3;
            __m128i v_bgr = _mm_loadu_si128((const __m128i*)p);

            __m128i r_u8 = _mm_shuffle_epi8(v_bgr, mask_r);
            __m128i g_u8 = _mm_shuffle_epi8(v_bgr, mask_g);
            __m128i b_u8 = _mm_shuffle_epi8(v_bgr, mask_b);

            __m128 r_f32 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(r_u8));
            __m128 g_f32 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(g_u8));
            __m128 b_f32 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b_u8));

            _mm_storeu_ps(dst_r + h * width + w, _mm_mul_ps(r_f32, v_inv255));
            _mm_storeu_ps(dst_g + h * width + w, _mm_mul_ps(g_f32, v_inv255));
            _mm_storeu_ps(dst_b + h * width + w, _mm_mul_ps(b_f32, v_inv255));
        }

        for (; w < width; ++w) {
            const uint8_t* p = row_ptr + w * 3;
            dst_r[h * width + w] = p[2] * kInv255;
            dst_g[h * width + w] = p[1] * kInv255;
            dst_b[h * width + w] = p[0] * kInv255;
        }
    }
}

/**
 * @brief Update best scores and class IDs branchlessly using SSE4.1.
 */
inline void update_best_scores_sse41(const float* current_scores, float* best_scores, int* best_class_ids, int class_id, int n) {
    __m128 v_class_id = _mm_castsi128_ps(_mm_set1_epi32(class_id));
    for (int i = 0; i <= n - 4; i += 4) {
        __m128 v_curr = _mm_loadu_ps(current_scores + i);
        __m128 v_best = _mm_loadu_ps(best_scores + i);
        
        __m128 v_mask = _mm_cmpgt_ps(v_curr, v_best);
        
        __m128 v_new_best = _mm_blendv_ps(v_best, v_curr, v_mask);
        _mm_storeu_ps(best_scores + i, v_new_best);
        
        __m128i v_best_ids = _mm_loadu_si128((__m128i*)(best_class_ids + i));
        __m128i v_new_ids = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(v_best_ids), v_class_id, v_mask));
        _mm_storeu_si128((__m128i*)(best_class_ids + i), v_new_ids);
    }
    
    for (int i = (n & ~3); i < n; ++i) {
        if (current_scores[i] > best_scores[i]) {
            best_scores[i] = current_scores[i];
            best_class_ids[i] = class_id;
        }
    }
}

/**
 * @brief Fast check if any of the 4 scores in a vector are above threshold.
 */
inline int check_threshold_sse41(const float* scores, float threshold) {
    __m128 v_scores = _mm_loadu_ps(scores);
    __m128 v_thresh = _mm_set1_ps(threshold);
    __m128 v_mask = _mm_cmpgt_ps(v_scores, v_thresh);
    return _mm_movemask_ps(v_mask);
}

} // namespace simd
