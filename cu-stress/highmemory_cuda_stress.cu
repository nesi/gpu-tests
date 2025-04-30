#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

// Kernel to maintain high compute intensity
__global__ void stressKernel(float *d_data, size_t n, unsigned long long iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = d_data[idx];
        float b = a;
        
        // Perform compute-heavy operations to maximize GPU usage
        for (unsigned long long i = 0; i < iterations; i++) {
            b = sinf(a) * cosf(a) / (sinf(a) * sinf(a) + cosf(a) * cosf(a));
            a = sinf(b) * cosf(b) * tanf(b) * expf(fabsf(sinf(b * 0.5f)));
            
            // Add more transcendental functions to increase compute load
            a = powf(a, 0.5f) * logf(fabsf(a) + 1.0f) * expf(sinf(a));
            b = tanf(a) * sqrtf(fabsf(a)) * cosf(b * a);
        }
        
        // Store result to prevent compiler optimization from removing the computation
        d_data[idx] = a + b;
    }
}

// Kernel to maximize memory bandwidth usage
__global__ void memoryStressKernel(float *d_in, float *d_out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Read and write operations to stress memory bandwidth
        float val = d_in[idx];
        val = val * 1.01f + 0.01f;
        d_out[idx] = val;
    }
}

int main(int argc, char *argv[]) {
    // Default test duration: 5 minutes (300 seconds)
    float durationMinutes = 5.0f;
    unsigned long long iterationsPerKernel = 5000; // Adjust based on testing
    
    // Parse command line arguments for duration if provided
    if (argc > 1) {
        durationMinutes = atof(argv[1]);
        if (durationMinutes < 0.1f) durationMinutes = 0.1f; // Minimum 6 seconds
    }
    
    // Parse iterations if provided
    if (argc > 2) {
        iterationsPerKernel = strtoull(argv[2], NULL, 10);
        if (iterationsPerKernel < 1000) iterationsPerKernel = 1000;
    }
    
    printf("Running H100 CUDA stress test for %.2f minutes with %llu iterations per kernel launch\n", 
           durationMinutes, iterationsPerKernel);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Global memory: %.2f GB\n", (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    
    // Calculate optimal execution configuration for H100
    int blockSize = 1024; // Maximum threads per block for H100
    size_t elementCount = prop.multiProcessorCount * 32 * blockSize; // Aiming for high occupancy
    
    // Allocate memory - adjust memory usage as desired
    // Increase this percentage (e.g., 0.1 = 10%, 0.5 = 50%, 0.8 = 80%) to use more GPU memory
    float memoryUsagePercentage = 0.5; // Using 50% of available memory - modify this value
    
    size_t availableMem = prop.totalGlobalMem * memoryUsagePercentage;
    size_t dataSize = (availableMem / 3) / sizeof(float); // Divide by 3 for input, output, and extra buffer
    
    if (dataSize < elementCount) {
        elementCount = dataSize;
    }
    
    printf("Memory usage percentage: %.1f%%\n", memoryUsagePercentage * 100.0f);
    
    size_t memSize = elementCount * sizeof(float);
    printf("Using %.2f GB of GPU memory\n", (float)(memSize * 3) / (1024.0f * 1024.0f * 1024.0f));
    
    // Host memory allocation
    float *h_data = (float*)malloc(memSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    
    // Initialize host data
    for (size_t i = 0; i < elementCount; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    
    // Device memory allocation
    float *d_data1, *d_data2, *d_data3;
    cudaMalloc((void**)&d_data1, memSize);
    cudaMalloc((void**)&d_data2, memSize);
    cudaMalloc((void**)&d_data3, memSize);
    
    // Copy data from host to device
    cudaMemcpy(d_data1, h_data, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data, memSize, cudaMemcpyHostToDevice);
    
    // Calculate grid size
    int gridSize = (elementCount + blockSize - 1) / blockSize;
    printf("Grid size: %d blocks, Block size: %d threads\n", gridSize, blockSize);
    
    // Set device to max performance
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    // Try to set maximum clock speeds if available
    printf("Attempting to set maximum clock speeds...\n");
    system("nvidia-smi -ac 1593,1980"); // H100 clocks - may require admin rights
    
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    float durationSeconds = durationMinutes * 60.0f;
    
    printf("Test started at: %s\n", __TIME__);
    printf("Running for %.1f seconds...\n", durationSeconds);
    
    // Main test loop
    int iteration = 0;
    while (true) {
        // Launch compute-intensive kernel
        stressKernel<<<gridSize, blockSize>>>(d_data1, elementCount, iterationsPerKernel);
        
        // Launch memory-intensive kernel
        memoryStressKernel<<<gridSize, blockSize>>>(d_data1, d_data2, elementCount);
        memoryStressKernel<<<gridSize, blockSize>>>(d_data2, d_data3, elementCount);
        memoryStressKernel<<<gridSize, blockSize>>>(d_data3, d_data1, elementCount);
        
        // Simple progress update
        if (iteration % 10 == 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float elapsedSeconds = std::chrono::duration<float>(currentTime - startTime).count();
            printf("\rProgress: %.1f%% (%.1f seconds elapsed)", 
                   (elapsedSeconds / durationSeconds) * 100.0f, elapsedSeconds);
            fflush(stdout);
            
            // Check if we've reached the desired duration
            if (elapsedSeconds >= durationSeconds) {
                break;
            }
        }
        
        iteration++;
    }
    
    // Calculate final stats
    auto endTime = std::chrono::high_resolution_clock::now();
    float totalSeconds = std::chrono::duration<float>(endTime - startTime).count();
    
    printf("\n\nTest completed successfully!\n");
    printf("Total test duration: %.2f seconds (%.2f minutes)\n", 
           totalSeconds, totalSeconds / 60.0f);
    printf("Completed %d kernel launches\n", iteration * 4); // 4 kernels per iteration
    
    // Copy result back (just to verify everything is working)
    cudaMemcpy(h_data, d_data1, memSize, cudaMemcpyDeviceToHost);
    
    // Calculate checksum for validation
    float checksum = 0.0f;
    for (size_t i = 0; i < elementCount; i++) {
        checksum += h_data[i];
    }
    printf("Data checksum: %f\n", checksum);
    
    // Free memory
    free(h_data);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    
    // Reset device
    cudaDeviceReset();
    
    return 0;
}
