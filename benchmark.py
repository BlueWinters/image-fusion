
import cv2
import numpy as np
import numba
import time
import fusion_cython


def functionWrapper(function, prefix):
    def callFunction(*args, **kwargs):
        beg = time.time()
        for n in range(1000):
            output = function(*args, **kwargs)
        end = time.time()
        eclipse = end - beg
        print('success call: {}({:.4f} ms)'.format(
            prefix, eclipse * 1000))
        return output
    return callFunction


def fuseImageByMaskWithNumpy(source_bgr, target_bgr, mask):
    multi = mask.astype(np.float32)[:, :, None] / 255.
    fusion = source_bgr * (1 - multi) + target_bgr * multi
    return np.round(fusion).astype(np.uint8)


@numba.jit(nopython=True, nogil=True, parallel=True)
def fuseImageByMaskWithNumba(matrix_a, matrix_b, matrix_c):
    result = np.empty_like(matrix_a)
    H, W, C = matrix_a.shape
    for i in numba.prange(H):
        for j in numba.prange(W):
            for c in numba.prange(C):
                result[i, j, c] = matrix_a[i, j, c] * (1 - matrix_c[i, j]) + matrix_b[i, j, c] * matrix_c[i, j]
    return result


if __name__ == '__main__':
    bgr = np.ascontiguousarray(cv2.imread('input.png'))
    mat = 255 - np.ascontiguousarray(cv2.imread('matting.png', -1))

    bgr = cv2.resize(bgr, (1400, 1400))
    mat = cv2.resize(mat, (1400, 1400))

    one = np.ascontiguousarray(np.ones_like(bgr)) * 255
    print(bgr.shape, mat.shape)

    out1 = functionWrapper(fusion_cython.fuseImageByMaskWithOpenMP, 'openmp')(bgr, one, mat)
    cv2.imwrite('output_openmp.png', out1)

    out2 = functionWrapper(fusion_cython.fuseImageByMaskWithSSE2, 'sse2')(bgr, one, mat)
    cv2.imwrite('output_sse2.png', out2)

    # out3 = functionWrapper(fusion_cython.fuseImageByMaskWithAVX2, 'avx2')(bgr, one, mat)
    # print(out3.shape)
    # cv2.imwrite('output_avx2.png', out3)

    bgr_t = np.ascontiguousarray(np.transpose(bgr, (2, 0, 1)))
    mat_t = np.ascontiguousarray(np.transpose(one, (2, 0, 1)))
    out3 = functionWrapper(fusion_cython.fuseImageByMaskWithAVX2, 'avx2')(bgr_t, mat_t, mat)
    cv2.imwrite('output_avx2.png', out3)

    # out4 = functionWrapper(fusion_cython.fuseImageByMaskWithAVX512, 'avx512')(bgr, one, mat)
    # cv2.imwrite('output_avx512.png', out4)

    out5 = functionWrapper(fuseImageByMaskWithNumpy, 'numpy')(bgr, one, mat)
    cv2.imwrite('output_numpy.png', out3)

    out6 = functionWrapper(fuseImageByMaskWithNumba, 'numba')(bgr, one, mat)
    cv2.imwrite('output_numba.png', out6)

    out7 = functionWrapper(fusion_cython.fuseImageByMaskWithCpp, 'cpp')(bgr, one, mat)
    cv2.imwrite('output_cpp.png', out7)
