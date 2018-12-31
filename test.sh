#!/bin/bash
make clean
make
mpirun -np 3 ./InnerProduct 9 3
