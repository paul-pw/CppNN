#!/bin/sh

./build.sh
gprofng collect app ./build/Release/src/mnist_nn  ./mnist-configs/test.config
gprofng display text -metrics name:i.%totalcpu:e.%totalcpu -limit 100 -functions test.1.er
