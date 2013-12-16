#!/bin/bash
cc train_cpu.c -o train_cpu -lm
cc train_gpu_jacob.c -o train_gpu_jacob -lOpenCL -lm -I/usr/local/cuda/include

