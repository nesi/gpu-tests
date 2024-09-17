#!/usr/bin/env python3

import torch
import time
import os
import multiprocessing

def create_large_tensor(size_gb, device):
    # Calculate number of elements for a given size in GB
    num_elements = int(size_gb * 1024 * 1024 * 1024 / 4)  # Assuming float32
    return torch.rand(num_elements, device=device)

def perform_computations(tensor, iterations):
    chunk_size = 1000000  # Adjust this value if needed
    for _ in range(iterations):
        # Process the tensor in chunks
        for i in range(0, tensor.size(0), chunk_size):
            chunk = tensor[i:i+chunk_size]
            # Perform computations on the chunk
            chunk = torch.sin(chunk)
            chunk = torch.exp(chunk)
            tensor[i:i+chunk_size] = chunk
    return tensor

def io_operations(file_size_gb, num_operations):
    filename = f"test_file_{multiprocessing.current_process().name}.bin"
    
    # Generate random data
    data = os.urandom(int(file_size_gb * 1024 * 1024 * 1024))
    
    for _ in range(num_operations):
        # Write data
        with open(filename, 'wb') as f:
            f.write(data)
        
        # Read data
        with open(filename, 'rb') as f:
            _ = f.read()
    
    # Clean up
    os.remove(filename)

def gpu_worker(gpu_id, run_time_minutes, gpu_memory_usage_gb, io_file_size_gb, io_operations_count):
    device = torch.device(f'cuda:{gpu_id}')
    
    print(f"Starting work on GPU {gpu_id}")
    
    # Create a large tensor to occupy GPU memory
    large_tensor = create_large_tensor(gpu_memory_usage_gb, device)
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < run_time_minutes * 60:
        # Perform GPU computations
        large_tensor = perform_computations(large_tensor, 10)
        
        # Perform I/O operations every 10 iterations
        if iteration % 10 == 0:
            io_operations(io_file_size_gb, io_operations_count)
        
        iteration += 1
    
    end_time = time.time()
    
    print(f"GPU {gpu_id} completed {iteration} iterations in {end_time - start_time:.2f} seconds")

def main():
    # Parameters
    num_gpus = 4
    gpu_memory_usage_gb = 72  # Aiming for 72GB usage per GPU
    run_time_minutes = 5
    io_file_size_gb = 10
    io_operations_count = 50

    print(f"Starting A100 GPU test on {num_gpus} GPUs...")
    
    # Create a process for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=gpu_worker, 
                                    args=(gpu_id, run_time_minutes, gpu_memory_usage_gb, 
                                          io_file_size_gb, io_operations_count))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All GPU tests completed")

if __name__ == "__main__":
    main()