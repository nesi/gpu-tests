#!/bin/bash

echo "===== Environment Information ====="
echo "CUDA driver version:"
nvidia-smi | grep "CUDA Version"
echo "Module list:"
module list
echo "===== End Environment Information ====="

# Check if the source file exists
echo "Looking for source file..."
if [ -f "highmemory_cuda_stress.cu" ]; then
    SOURCE_FILE="highmemory_cuda_stress.cu"
    echo "Found source file: cuda_stress.cu"
elif [ -f "highmemory_cuda_stress.cu" ]; then
    SOURCE_FILE="highmemory_cuda_stress.cu"
    echo "Found source file: cuda_stress.cu"
else
    echo "Error: Cannot find CUDA source file. Please make sure cuda_stress.cu or cuda_stress.cu exists."
    ls -la *.cu
    exit 1
fi

echo "===== First attempt: Using NVHPC compiler with explicit include paths ====="
# Try nvc++ with explicit include paths
NVHPC_ROOT=$(dirname $(dirname $(which nvc++)))
GCC_ROOT=$(dirname $(dirname $(which gcc)))

echo "NVHPC_ROOT: $NVHPC_ROOT"
echo "GCC_ROOT: $GCC_ROOT"
echo "Source file: $SOURCE_FILE"

# Use nvc++ but with explicit include paths to system headers
nvc++ -O3 -fast -cuda -gpu=cc90 \
    -I${NVHPC_ROOT}/include \
    -I${GCC_ROOT}/include \
    -I/usr/include \
    -o cuda_stress_test $SOURCE_FILE

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "NVHPC compilation failed! Trying with nvcc..."

    # Try direct nvcc compilation
    echo "===== Second attempt: Using NVCC directly ====="
    NVCC_PATH=$(which nvcc)
    echo "nvcc path: $NVCC_PATH"
    CUDA_PATH=$(dirname $(dirname $NVCC_PATH))
    echo "CUDA path: $CUDA_PATH"

    nvcc -O3 -arch=sm_90 \
        -I${CUDA_PATH}/include \
        -I/usr/include \
        -o cuda_stress_test $SOURCE_FILE

    if [ $? -ne 0 ]; then
        echo "===== Final attempt: Using NVCC with all available paths ====="
        # Get all potential include paths
        INCLUDE_DIRS=""
        for dir in /usr/include /usr/local/include ${CUDA_PATH}/include ${NVHPC_ROOT}/include ${GCC_ROOT}/include; do
            if [ -d "$dir" ]; then
                INCLUDE_DIRS="$INCLUDE_DIRS -I$dir"
            fi
        done

        echo "Using include directories: $INCLUDE_DIRS"
        nvcc -O3 -arch=sm_90 $INCLUDE_DIRS -o cuda_stress_test $SOURCE_FILE

        if [ $? -ne 0 ]; then
            echo "All compilation attempts failed."
            echo "Let's print some diagnostic information:"
            echo "1. System headers location:"
            ls -la /usr/include/limits.h
            echo "2. Compiler search paths:"
            echo "NVCC include search:"
            nvcc --verbose --dryrun $SOURCE_FILE 2>&1 | grep -i "include"
            echo "NVC++ include search:"
            nvc++ --show-search-path
            exit 1
        fi
    fi
fi

# Final check to see if the executable was created
if [ -f "cuda_stress_test" ]; then
    echo "Compilation successful. CUDA stress test is ready to run."
    ls -la cuda_stress_test
else
    echo "Compilation appeared to succeed but executable 'cuda_stress_test' was not created!"
    echo "Listing current directory contents:"
    ls -la
    exit 1
fi
