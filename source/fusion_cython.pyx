import numpy as np
cimport numpy as np

np.import_array()  # 初始化 numpy C-API

cdef extern from "fusion.h":
    void fuseImageByMask_Standard(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* output,
        int h, int w, int c
    );

    void fuseImageByMask_OpenMP(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* out,
        int h, int w, int c
    );

    void fuseImageByMask_SSE2(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* out,
        int h, int w, int c
    );

    void fuseImageByMask_AVX2(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* output,
        int h, int w, int c
    );

def fuseImageByMaskWithCpp(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int h = source_bgr.shape[0]
    cdef int w = source_bgr.shape[1]
    cdef int c = source_bgr.shape[2]
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((h, w, c), dtype=np.uint8)
    fuseImageByMask_Standard(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        h, w, c
    )
    return fusion

def fuseImageByMaskWithOpenMP(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int h = source_bgr.shape[0]
    cdef int w = source_bgr.shape[1]
    cdef int c = source_bgr.shape[2]
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((h, w, c), dtype=np.uint8)
    fuseImageByMask_OpenMP(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        h, w, c
    )
    return fusion

def fuseImageByMaskWithSSE2(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int h = source_bgr.shape[0]
    cdef int w = source_bgr.shape[1]
    cdef int c = source_bgr.shape[2]
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((h, w, c), dtype=np.uint8)
    fuseImageByMask_SSE2(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        h, w, c
    )
    return fusion

def fuseImageByMaskWithAVX2(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int h = source_bgr.shape[1]
    cdef int w = source_bgr.shape[2]
    cdef int c = source_bgr.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((c, h, w), dtype=np.uint8)
    fuseImageByMask_AVX2(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        h, w, c
    )
    return np.transpose(fusion, (1,2,0))

