# lifted-eigen

## compile
compile on mac:
g++ -std=c++11 lifted_eigen_test.cpp -lcholmod -O3 -o lifted

compile on max (clang++):
clang++ lifted_eigen_test.cpp -I/usr/local/include -fopenmp -L/usr/local/opt/llvm/lib -L/usr/local/lib -lcholmod -O3 -o lifted


## usage
./lifted
