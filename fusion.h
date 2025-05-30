#ifndef __Fusion__
#define __Fusion__


void fuseImageByMask_Standard(
    const unsigned char* source,
    const unsigned char* target,
    const unsigned char* mask,
    unsigned char* output,
    int h, int w, int c
);

void fuseImageByMask_OpenMP(
    const unsigned char* src,
    const unsigned char* blur,
    const unsigned char* mask,
    unsigned char* out,
    int h, int w, int c
);

void fuseImageByMask_SSE2(
    const unsigned char* src,
    const unsigned char* blur,
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

void fuseImageByMask_AVX512(
    const unsigned char* source,
    const unsigned char* target,
    const unsigned char* mask,
    unsigned char* output,
    int h, int w, int c
);

#endif