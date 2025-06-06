import numpy as np
cimport numpy as np

np.import_array()  # 初始化 numpy C-API

cdef extern from "fusion.h":
    int Fusion_Error
    int Fusion_Native
    int Fusion_OpenMP
    int Fusion_OpenMP_SSE2
    int Fusion_OpenMP_AVX2

    int fuseImageByMask_Native(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* output,
        int h, int w, int c
    );

    int fuseImageByMask_OpenMP(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* out,
        int h, int w, int c
    );

    int fuseImageByMask_SSE2(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* out,
        int h, int w, int c
    );

    int fuseImageByMask_AVX2_CHW(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* output,
        int h, int w, int c
    );

    int fuseImageByMask_HWC(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* output,
        int h, int w, int c
    );

    int fuseImageByMask_CHW(
        const unsigned char* source,
        const unsigned char* target,
        const unsigned char* mask,
        unsigned char* output,
        int h, int w, int c
    );



def fuseImage_Native(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int src_h = source_bgr.shape[0]
    cdef int src_w = source_bgr.shape[1]
    cdef int src_c = source_bgr.shape[2]
    cdef int tar_h = target_bgr.shape[0]
    cdef int tar_w = target_bgr.shape[1]
    cdef int tar_c = target_bgr.shape[2]
    if src_h != tar_h or src_w != tar_w:
        raise ValueError("source_bgr and target_bgr shape mismatch")
    if src_c != 3 or tar_c != 3:
        raise ValueError("source_bgr or target_bgr must have 3 channels") 
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((src_h, src_w, src_c), dtype=np.uint8)
    cdef int flag = fuseImageByMask_Native(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        src_h, src_w, src_c
    )
    return flag, fusion

def fuseImage_OpenMP(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int src_h = source_bgr.shape[0]
    cdef int src_w = source_bgr.shape[1]
    cdef int src_c = source_bgr.shape[2]
    cdef int tar_h = target_bgr.shape[0]
    cdef int tar_w = target_bgr.shape[1]
    cdef int tar_c = target_bgr.shape[2]
    if src_h != tar_h or src_w != tar_w:
        raise ValueError("source_bgr and target_bgr shape mismatch")
    if src_c != 3 or tar_c != 3:
        raise ValueError("source_bgr or target_bgr must have 3 channels") 
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((src_h, src_w, src_c), dtype=np.uint8)
    cdef int flag = fuseImageByMask_OpenMP(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        src_h, src_w, src_c
    )
    return flag, fusion

def fuseImage_SSE2(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int src_h = source_bgr.shape[0]
    cdef int src_w = source_bgr.shape[1]
    cdef int src_c = source_bgr.shape[2]
    cdef int tar_h = target_bgr.shape[0]
    cdef int tar_w = target_bgr.shape[1]
    cdef int tar_c = target_bgr.shape[2]
    if src_h != tar_h or src_w != tar_w:
        raise ValueError("source_bgr and target_bgr shape mismatch")
    if src_c != 3 or tar_c != 3:
        raise ValueError("source_bgr or target_bgr must have 3 channels") 
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((src_h, src_w, src_c), dtype=np.uint8)
    cdef int flag = fuseImageByMask_SSE2(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        src_h, src_w, src_c
    )
    return flag, fusion

def fuseImage_AVX2_CHW(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int src_h = source_bgr.shape[1]
    cdef int src_w = source_bgr.shape[2]
    cdef int src_c = source_bgr.shape[0]
    cdef int tar_h = target_bgr.shape[1]
    cdef int tar_w = target_bgr.shape[2]
    cdef int tar_c = target_bgr.shape[0]
    if src_h != tar_h or src_w != tar_w:
        raise ValueError("source_bgr and target_bgr shape mismatch")
    if src_c != 3 or tar_c != 3:
        raise ValueError("source_bgr or target_bgr must have 3 channels") 
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((src_c, src_h, src_w), dtype=np.uint8)
    cdef int flag = fuseImageByMask_AVX2_CHW(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        src_h, src_w, src_c
    )
    return flag, fusion

def fuseImage_HWC(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int src_h = source_bgr.shape[0]
    cdef int src_w = source_bgr.shape[1]
    cdef int src_c = source_bgr.shape[2]
    cdef int tar_h = target_bgr.shape[0]
    cdef int tar_w = target_bgr.shape[1]
    cdef int tar_c = target_bgr.shape[2]
    if src_h != tar_h or src_w != tar_w:
        raise ValueError("source_bgr and target_bgr shape mismatch")
    if src_c != 3 or tar_c != 3:
        raise ValueError("source_bgr or target_bgr must have 3 channels") 
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((src_h, src_w, src_c), dtype=np.uint8)
    cdef int flag = fuseImageByMask_HWC(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        src_h, src_w, src_c
    )
    return flag, fusion

def fuseImage_CHW(
    np.ndarray[np.uint8_t, ndim=3] source_bgr,
    np.ndarray[np.uint8_t, ndim=3] target_bgr,
    np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int src_h = source_bgr.shape[1]
    cdef int src_w = source_bgr.shape[2]
    cdef int src_c = source_bgr.shape[0]
    cdef int tar_h = target_bgr.shape[1]
    cdef int tar_w = target_bgr.shape[2]
    cdef int tar_c = target_bgr.shape[0]
    if src_h != tar_h or src_w != tar_w:
        raise ValueError("source_bgr and target_bgr shape mismatch")
    if src_c != 3 or tar_c != 3:
        raise ValueError("source_bgr or target_bgr must have 3 channels") 
    cdef np.ndarray[np.uint8_t, ndim=3] fusion = np.empty((src_c, src_h, src_w), dtype=np.uint8)
    cdef int flag = fuseImageByMask_CHW(
        <const unsigned char*>source_bgr.data,
        <const unsigned char*>target_bgr.data,
        <const unsigned char*>mask.data,
        <unsigned char*>fusion.data,
        src_h, src_w, src_c
    )
    return flag, fusion


Fusion_Result_Error = Fusion_Error
Fusion_Result_Native = Fusion_Native
Fusion_Result_OpenMP = Fusion_OpenMP
Fusion_Result_OpenMP_SSE2 = Fusion_OpenMP_SSE2
Fusion_Result_OpenMP_AVX2 = Fusion_OpenMP_AVX2
