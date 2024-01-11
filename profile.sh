#!/bin/sh

cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_BUILD_TYPE=RelWithDebInfo -S . -B build/RelProf
cmake --build build/RelProf
./build/RelProf/src/mnist_nn ./mnist-configs/test.config 
gprof ./build/RelProf/src/mnist_nn gmon.out > analysis.txt
