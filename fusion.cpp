#include <math.h>
#include <stdio.h>
#include "fusion.h"

void fuseImageByMask_Standard(
    const unsigned char* source,
    const unsigned char* target,
    const unsigned char* mask,
    unsigned char* output,
    int h, int w, int c
)
{
    int i, j, idx;
    float m, inv_m;
    for (i = 0; i < h; ++i)
    {
        for (j = 0; j < w; ++j)
        {
            m = mask[i * w + j] / 255.0f;
            inv_m = 1.0f - m;
            idx = (i * w + j) * c;
            output[idx + 0] = (unsigned char)roundf(source[idx + 0] * inv_m + target[idx + 0] * m);
            output[idx + 1] = (unsigned char)roundf(source[idx + 1] * inv_m + target[idx + 1] * m);
            output[idx + 2] = (unsigned char)roundf(source[idx + 2] * inv_m + target[idx + 2] * m);
        }
    }
}


void fuseImageByMask_OpenMP(
    const unsigned char* source,
    const unsigned char* target,
    const unsigned char* mask,
    unsigned char* output,
    int h, int w, int c
)
{
    int i, j, idx;
    float m, inv_m;
    #pragma omp parallel for private(j, idx, m, inv_m)
    for (i = 0; i < h; ++i)
    {
        for (j = 0; j < w; ++j)
        {
            m = mask[i * w + j] / 255.0f;
            inv_m = 1.0f - m;
            idx = (i * w + j) * c;
            output[idx + 0] = (unsigned char)roundf(source[idx + 0] * inv_m + target[idx + 0] * m);
            output[idx + 1] = (unsigned char)roundf(source[idx + 1] * inv_m + target[idx + 1] * m);
            output[idx + 2] = (unsigned char)roundf(source[idx + 2] * inv_m + target[idx + 2] * m);
        }
    }
}


#include <emmintrin.h> // SSE2
#include <xmmintrin.h>
#include <math.h>
#include <stdint.h>

void fuseImageByMask_SSE2(
    const unsigned char* source,
    const unsigned char* target,
    const unsigned char* mask,
    unsigned char* output,
    int h, int w, int c
)
{
    int size = h * w;
    if (c != 3) {
        // 回退到普通实现
        fuseImageByMask_OpenMP(source, target, mask, output, h, w, c);
        return;
    }
    #pragma omp parallel for
    for (int i = 0; i < size - 3; i += 4) {
        __m128 m = _mm_set_ps(
            mask[i + 3] / 255.0f,
            mask[i + 2] / 255.0f,
            mask[i + 1] / 255.0f,
            mask[i + 0] / 255.0f
        );
        __m128 inv_m = _mm_sub_ps(_mm_set1_ps(1.0f), m);
        for (int k = 0; k < 3; ++k) {
            __m128 src = _mm_set_ps(
                source[(i + 3) * 3 + k],
                source[(i + 2) * 3 + k],
                source[(i + 1) * 3 + k],
                source[(i + 0) * 3 + k]
            );
            __m128 tgt = _mm_set_ps(
                target[(i + 3) * 3 + k],
                target[(i + 2) * 3 + k],
                target[(i + 1) * 3 + k],
                target[(i + 0) * 3 + k]
            );
            __m128 val = _mm_add_ps(_mm_mul_ps(src, inv_m), _mm_mul_ps(tgt, m));
            val = _mm_add_ps(val, _mm_set1_ps(0.5f));
            __m128i ival = _mm_cvtps_epi32(val);
            output[(i + 0) * 3 + k] = (unsigned char)(_mm_extract_epi16(ival, 0) & 0xFF);
            output[(i + 1) * 3 + k] = (unsigned char)(_mm_extract_epi16(ival, 2) & 0xFF);
            output[(i + 2) * 3 + k] = (unsigned char)(_mm_extract_epi16(ival, 4) & 0xFF);
            output[(i + 3) * 3 + k] = (unsigned char)(_mm_extract_epi16(ival, 6) & 0xFF);
        }
    }
    // 处理剩余像素
    for (int i = (size & ~3); i < size; ++i) {
        float m = mask[i] / 255.0f;
        float inv_m = 1.0f - m;
        int idx = i * 3;
        output[idx + 0] = (unsigned char)roundf(source[idx + 0] * inv_m + target[idx + 0] * m);
        output[idx + 1] = (unsigned char)roundf(source[idx + 1] * inv_m + target[idx + 1] * m);
        output[idx + 2] = (unsigned char)roundf(source[idx + 2] * inv_m + target[idx + 2] * m);
    }
}


#include "fusion.h"
#include <math.h>
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
static inline bool is_avx2_supported() {
#ifdef __AVX2__
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
#else
static inline bool is_avx2_supported() { return false; }
#endif

void fuseImageByMask_AVX2(
    const unsigned char* source,
    const unsigned char* target,
    const unsigned char* mask,
    unsigned char* output,
    int h, int w, int c
)
{
    int size = h * w;
    if (c == 3) {
        const int step = 8; // AVX2一次处理8个像素
        int main_loop = size / step * step;

        const unsigned char* src_c0 = source;
        const unsigned char* src_c1 = source + size;
        const unsigned char* src_c2 = source + size * 2;
        const unsigned char* tgt_c0 = target;
        const unsigned char* tgt_c1 = target + size;
        const unsigned char* tgt_c2 = target + size * 2;
        unsigned char* out_c0 = output;
        unsigned char* out_c1 = output + size;
        unsigned char* out_c2 = output + size * 2;

        #pragma omp parallel for
        for (int i = 0; i < main_loop; i += step) {
            // mask
            __m128i mask_u8 = _mm_loadl_epi64((const __m128i*)(mask + i)); // 8字节
            __m256i mask_u32 = _mm256_cvtepu8_epi32(mask_u8);
            __m256 mask_f = _mm256_cvtepi32_ps(mask_u32);
            mask_f = _mm256_div_ps(mask_f, _mm256_set1_ps(255.0f));
            __m256 inv_mask_f = _mm256_sub_ps(_mm256_set1_ps(1.0f), mask_f);

            // 通道0
            __m256i src_u8_0 = _mm256_loadu_si256((const __m256i*)(src_c0 + i));
            __m256i tgt_u8_0 = _mm256_loadu_si256((const __m256i*)(tgt_c0 + i));
            __m256i src_u32_0 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(src_u8_0));
            __m256i tgt_u32_0 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(tgt_u8_0));
            __m256 src_f_0 = _mm256_cvtepi32_ps(src_u32_0);
            __m256 tgt_f_0 = _mm256_cvtepi32_ps(tgt_u32_0);
            __m256 val_f_0 = _mm256_add_ps(_mm256_mul_ps(src_f_0, inv_mask_f), _mm256_mul_ps(tgt_f_0, mask_f));
            val_f_0 = _mm256_add_ps(val_f_0, _mm256_set1_ps(0.5f));
            __m256i val_i_0 = _mm256_cvtps_epi32(val_f_0);
            __m128i val_u8_0 = _mm_packus_epi16(
                _mm_packus_epi32(_mm256_extracti128_si256(val_i_0, 0), _mm256_extracti128_si256(val_i_0, 1)),
                _mm_setzero_si128());
            _mm_storel_epi64((__m128i*)(out_c0 + i), val_u8_0);

            // 通道1
            __m256i src_u8_1 = _mm256_loadu_si256((const __m256i*)(src_c1 + i));
            __m256i tgt_u8_1 = _mm256_loadu_si256((const __m256i*)(tgt_c1 + i));
            __m256i src_u32_1 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(src_u8_1));
            __m256i tgt_u32_1 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(tgt_u8_1));
            __m256 src_f_1 = _mm256_cvtepi32_ps(src_u32_1);
            __m256 tgt_f_1 = _mm256_cvtepi32_ps(tgt_u32_1);
            __m256 val_f_1 = _mm256_add_ps(_mm256_mul_ps(src_f_1, inv_mask_f), _mm256_mul_ps(tgt_f_1, mask_f));
            val_f_1 = _mm256_add_ps(val_f_1, _mm256_set1_ps(0.5f));
            __m256i val_i_1 = _mm256_cvtps_epi32(val_f_1);
            __m128i val_u8_1 = _mm_packus_epi16(
                _mm_packus_epi32(_mm256_extracti128_si256(val_i_1, 0), _mm256_extracti128_si256(val_i_1, 1)),
                _mm_setzero_si128());
            _mm_storel_epi64((__m128i*)(out_c1 + i), val_u8_1);

            // 通道2
            __m256i src_u8_2 = _mm256_loadu_si256((const __m256i*)(src_c2 + i));
            __m256i tgt_u8_2 = _mm256_loadu_si256((const __m256i*)(tgt_c2 + i));
            __m256i src_u32_2 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(src_u8_2));
            __m256i tgt_u32_2 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(tgt_u8_2));
            __m256 src_f_2 = _mm256_cvtepi32_ps(src_u32_2);
            __m256 tgt_f_2 = _mm256_cvtepi32_ps(tgt_u32_2);
            __m256 val_f_2 = _mm256_add_ps(_mm256_mul_ps(src_f_2, inv_mask_f), _mm256_mul_ps(tgt_f_2, mask_f));
            val_f_2 = _mm256_add_ps(val_f_2, _mm256_set1_ps(0.5f));
            __m256i val_i_2 = _mm256_cvtps_epi32(val_f_2);
            __m128i val_u8_2 = _mm_packus_epi16(
                _mm_packus_epi32(_mm256_extracti128_si256(val_i_2, 0), _mm256_extracti128_si256(val_i_2, 1)),
                _mm_setzero_si128());
            _mm_storel_epi64((__m128i*)(out_c2 + i), val_u8_2);
        }
        // 处理剩余像素
        for (int i = main_loop; i < size; ++i) {
            float m = mask[i] / 255.0f;
            float inv_m = 1.0f - m;
            out_c0[i] = (unsigned char)roundf(src_c0[i] * inv_m + tgt_c0[i] * m);
            out_c1[i] = (unsigned char)roundf(src_c1[i] * inv_m + tgt_c1[i] * m);
            out_c2[i] = (unsigned char)roundf(src_c2[i] * inv_m + tgt_c2[i] * m);
        }
        return;
    }
}
