# floatQuantization
Qunatization examples with float 32 to 16 to 8

Inspiration/code adapted from https://hbfs.wordpress.com/2013/02/12/float16/

The cuda examples require cuda 9.2. Beforehand the float16 code is marked as `_device_` only.

To compile and run the `unions.cpp` examples do:
```bash
g++ unions.cpp
./a.out
```
To compile and run the MPI examples:
```bash
mpic++ unions.cpp -DUSE_MPI
mpiexec -np 2 --tag-output ./a.out
```
