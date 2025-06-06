# Image-Fusion
Image fusion implementation in C++ and assembly

## Performance


<div align="center">


| method| implement     | time (ns) |
|:------|:--------------|:----------|
| openmp| native openmp | 391       |
| sse2  | sse2+openmp   | 177       |
| avx2  | avx2+openmp   | 41        |
| numpy | pure numpy    | 8916      |
| numba | numpy+numba   | 5406      |
| native| native cpp    | 3279      |


</div>

