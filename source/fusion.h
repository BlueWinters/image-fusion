#ifndef __Fusion__
#define __Fusion__

#define Fusion_Error               0
#define Fusion_Native              1
#define Fusion_OpenMP              2
#define Fusion_OpenMP_SSE2         3
#define Fusion_OpenMP_AVX2         4
#define Fusion_OpenMP_AVX512       5


int fuseImageByMask_Native(
    const unsigned char* source,
    const unsigned char* target,
    const unsigned char* mask,
    unsigned char* output,
    int h, int w, int c
);

int fuseImageByMask_OpenMP(
    const unsigned char* src,
    const unsigned char* blur,
    const unsigned char* mask,
    unsigned char* out,
    int h, int w, int c
);

int fuseImageByMask_SSE2(
    const unsigned char* src,
    const unsigned char* blur,
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

#endif