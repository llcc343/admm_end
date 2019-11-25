#!/bin/bash

cd cmake-build-debug/
cmake ..
make
cd ../bin
mpirun -np 8 ./admm_end 4 ../../../../dataSet/rcv1/%d/data_%d 7 1