# Image-Fusion
Image fusion implementation in C++ and assembly

## Build
```python
python setup.py build_ext --inplace
```

## Performance
- note: the resolution of the test image is 700x700 which can be found in `fold/asset`
- run the following code to test the performance: 
```python
python benchmark.py
```


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

