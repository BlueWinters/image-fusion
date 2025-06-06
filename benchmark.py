
import cv2
import numpy as np
import numba
import time
import fusion_cython


def functionWrapper(function, prefix):
    def callFunction(*args, **kwargs):
        beg = time.time()
        for n in range(10):
            output = function(*args, **kwargs)
        end = time.time()
        eclipse = end - beg
        print('success call: {}({}, {:.4f} ms)'.format(
            prefix, output[0], eclipse * 1000))
        return output[1]
    return callFunction


def fuseImage_Numpy(source_bgr, target_bgr, mask):
    multi = mask.astype(np.float32)[:, :, None] / 255.
    fusion = source_bgr * multi + target_bgr * (1 - multi)
    return np.round(fusion).astype(np.uint8)


@numba.jit(nopython=True, nogil=True, parallel=True)
def fuseImage_Numba(matrix_a, matrix_b, matrix_c):
    result = np.empty_like(matrix_a)
    H, W, C = matrix_a.shape
    for i in numba.prange(H):
        for j in numba.prange(W):
            for c in numba.prange(C):
                result[i, j, c] = matrix_a[i, j, c] * matrix_c[i, j] + matrix_b[i, j, c] * (1 - matrix_c[i, j])
    return result


if __name__ == '__main__':
    bgr = np.ascontiguousarray(cv2.imread('input.png'))
    mat = np.ascontiguousarray(cv2.imread('alpha.png', -1))

    # bgr = cv2.resize(bgr, (1400, 1400))
    # mat = cv2.resize(mat, (1400, 1400))

    one = np.ascontiguousarray(np.ones_like(bgr)) * 255
    print(bgr.shape, mat.shape)

    out1 = functionWrapper(fusion_cython.fuseImage_OpenMP, 'openmp')(bgr, one, mat)
    cv2.imwrite('output_openmp.png', out1)

    out2 = functionWrapper(fusion_cython.fuseImage_SSE2, 'sse2')(bgr[:, :, :], one, mat)
    cv2.imwrite('output_sse2.png', out2)

    out3 = functionWrapper(fusion_cython.fuseImageWithAVX2, 'avx2')(bgr, one, mat)
    cv2.imwrite('output_avx2.png', out3)

    bgr_t = np.ascontiguousarray(np.transpose(bgr, (2, 0, 1)))
    mat_t = np.ascontiguousarray(np.transpose(one, (2, 0, 1)))
    out3 = functionWrapper(fusion_cython.fuseImage_AVX2_CHW, 'avx2')(bgr_t, mat_t, mat)
    cv2.imwrite('output_avx2.png', np.transpose(out3, (1,2,0)))

    out4 = functionWrapper(fusion_cython.fuseImage_AVX512, 'avx512')(bgr, one, mat)
    cv2.imwrite('output_avx512.png', out4)

    out5 = functionWrapper(fuseImage_Numpy, 'numpy')(bgr, one, mat)
    cv2.imwrite('output_numpy.png', out3)

    out6 = functionWrapper(fuseImage_Numba, 'numba')(bgr, one, mat)
    cv2.imwrite('output_numba.png', out6)

    out7 = functionWrapper(fusion_cython.fuseImage_Cpp, 'cpp')(bgr, one, mat)
    cv2.imwrite('output_cpp.png', out7)

    out8 = functionWrapper(fusion_cython.fuseImage_HWC, 'hwc')(bgr, one, mat)
    cv2.imwrite('output_hwc.png', out8)

    bgr_t = np.ascontiguousarray(np.transpose(bgr, (2, 0, 1)))
    mat_t = np.ascontiguousarray(np.transpose(one, (2, 0, 1)))
    out9 = functionWrapper(fusion_cython.fuseImage_CHW, 'chw')(bgr_t, mat_t, mat)
    cv2.imwrite('output_chw.png', np.transpose(out9, (1, 2, 0)))

    print(fusion_cython.Fusion_Result_Error)
    print(fusion_cython.Fusion_Result_Native)
    print(fusion_cython.Fusion_Result_OpenMP)
    print(fusion_cython.Fusion_Result_OpenMP_SSE2)
    print(fusion_cython.Fusion_Result_OpenMP_AVX2)
