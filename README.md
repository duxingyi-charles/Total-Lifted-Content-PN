# TLC-PN

TLC-PN computes locally injective mappings from a source mesh to a
user-specified target domain. The program recovers an injective mapping from a non-injective
initial embedding by minimizing the TLC (Total Lifted Content) energy proposed in our paper
[Lifting Simplices to Find Injectivity](https://duxingyi-charles.github.io/publication/lifting-simplices-to-find-injectivity/).

The TLC energy is minimized by projected Newton (PN) method.

Here is a similar program called [TLC-QN](https://github.com/duxingyi-charles/lifting_simplices_to_find_injectivity)
 based on quasi-Newton method.

## Dependency

Eigen and SuiteSparse, openMP

## compile
compile on mac:
g++ -std=c++11 lifted_eigen_test.cpp -lcholmod -O3 -o lifted

compile on max (clang++):
clang++ lifted_eigen_test.cpp -I/usr/local/include -fopenmp -L/usr/local/opt/llvm/lib -L/usr/local/lib -lcholmod -O3 -o lifted


## How to use
./lifted

## data format

## Cite

