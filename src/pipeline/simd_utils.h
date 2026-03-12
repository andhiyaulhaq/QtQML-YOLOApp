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
 * 
 * @param src Pointer to BGR image data (uint8_t)
 * @param dst Pointer to destination blob (float*)
 * @param width Image width
 * @param height Image height
 * @param step Number of bytes per row in source image
 */
inline void hwc_to_chw_bgr_to_rgb_sse41(const uint8_t* src, float* dst, int width, int height, int step) {
    const float kInv255 = 1.0f / 255.0f;
    const int plane_size = width * height;
    float* dst_r = dst;
    float* dst_g = dst + plane_size;
    float* dst_b = dst + 2 * plane_size;

    __m128 v_inv255 = _mm_set1_ps(kInv255);

    // Shuffle masks for de-interleaving 4 BGR pixels (12 bytes) from a 16-byte load
    // Source: [B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3 XX XX XX XX]
    __m128i mask_r = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i mask_g = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i mask_b = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    for (int h = 0; h < height; ++h) {
        const uint8_t* row_ptr = src + h * step;
        int w = 0;
        
        for (; w <= width - 4; w += 4) {
            const uint8_t* p = row_ptr + w * 3;
            __m128i v_bgr = _mm_loadu_si128((const __m128i*)p);

            // De-interleave using shuffles
            __m128i r_u8 = _mm_shuffle_epi8(v_bgr, mask_r);
            __m128i g_u8 = _mm_shuffle_epi8(v_bgr, mask_g);
            __m128i b_u8 = _mm_shuffle_epi8(v_bgr, mask_b);

            // Convert uint8 to float (first 4 elements)
            __m128 r_f32 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(r_u8));
            __m128 g_f32 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(g_u8));
            __m128 b_f32 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b_u8));

            // Normalize and store
            _mm_storeu_ps(dst_r + h * width + w, _mm_mul_ps(r_f32, v_inv255));
            _mm_storeu_ps(dst_g + h * width + w, _mm_mul_ps(g_f32, v_inv255));
            _mm_storeu_ps(dst_b + h * width + w, _mm_mul_ps(b_f32, v_inv255));
        }

        // Tail loop
        for (; w < width; ++w) {
            const uint8_t* p = row_ptr + w * 3;
            dst_r[h * width + w] = p[2] * kInv255;
            dst_g[h * width + w] = p[1] * kInv255;
            dst_b[h * width + w] = p[0] * kInv255;
        }
    }
}

} // namespace simd
