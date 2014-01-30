#!/bin/bash
g++ -O3 train_cpu.c -o train_cpu -lm
g++ -O3 -I/usr/local/cuda-5.5/targets/x86_64-linux/include train_gpu_jacob.c -o train_gpu_jacob -lOpenCL -lm 

