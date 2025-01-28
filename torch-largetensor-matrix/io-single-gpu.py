#!/usr/bin/env python3

import torch
import time
import os

def create_large_tensor(size_gb):
    """
    Creates a large tensor on GPU to simulate memory usage.
    Args:
        size_gb: Size of tensor in gigabytes
    Returns:
        torch.Tensor: Random tensor of specified size on GPU
    """
    # Calculate number of elements needed for the requested GB size
    # We divide by 4 because each float32 takes 4 bytes
    num_elements = int(size_gb * 1024 * 1024 * 1024 / 4)
    return torch.rand(num_elements, device='cuda')

def perform_computations(tensor, iterations):
    """
    Performs mathematical operations on the tensor in chunks to avoid OOM.
    Args:
        tensor: Input tensor to perform computations on
        iterations: Number of times to repeat computations
    Returns:
        torch.Tensor: Processed tensor
    """
    chunk_size = 1000000  # Process 1M elements at a time
    for _ in range(iterations):
        for i in range(0, tensor.size(0), chunk_size):
            chunk = tensor[i:i+chunk_size]
            # Chain of computations to stress GPU
            chunk = torch.sin(chunk)
            chunk = torch.exp(chunk)
            tensor[i:i+chunk_size] = chunk
    return tensor

def io_operations(file_size_gb, num_operations):
    """
    Performs intensive I/O operations and collects performance metrics.
    Args:
        file_size_gb: Size of test file in gigabytes
        num_operations: Number of write/read cycles to perform
    Returns:
        list: Collection of I/O performance metrics
    """
    filename = "test_file.bin"
    metrics = []
    
    # Generate random data once and reuse it
    data = os.urandom(int(file_size_gb * 1024 * 1024 * 1024))
    
    for op in range(num_operations):
        start_time = time.time()
        
        # Write data
        write_start = time.time()
        with open(filename, 'wb') as f:
            f.write(data)
        write_end = time.time()
        
        # Read data
        read_start = time.time()
        with open(filename, 'rb') as f:
            _ = f.read()
        read_end = time.time()
        
        # Calculate metrics
        write_duration = write_end - write_start
        read_duration = read_end - read_start
        write_speed_gbps = file_size_gb / write_duration
        read_speed_gbps = file_size_gb / read_duration
        
        metrics.append({
            'operation': op,
            'write_duration': write_duration,
            'read_duration': read_duration,
            'write_speed_gbps': write_speed_gbps,
            'read_speed_gbps': read_speed_gbps,
            'total_duration': time.time() - start_time
        })
        
        try:
            os.remove(filename)
        except OSError as e:
            print(f"Cleanup failed: {e}")
            
    return metrics

def main():
    # Test parameters
    gpu_memory_usage_gb = 72  # Targeting 72GB GPU memory usage
    run_time_minutes = 5      # Total test duration
    io_file_size_gb = 10     # Size of I/O test file
    io_operations_count = 50  # Number of I/O operations per batch
    
    print("Starting A100 GPU test...")
    
    # Initialize GPU tensor
    large_tensor = create_large_tensor(gpu_memory_usage_gb)
    
    start_time = time.time()
    iteration = 0
    all_metrics = []
    
    while time.time() - start_time < run_time_minutes * 60:
        # GPU computations
        large_tensor = perform_computations(large_tensor, 10)
        
        # I/O operations every 10 iterations
        if iteration % 10 == 0:
            try:
                metrics = io_operations(io_file_size_gb, io_operations_count)
                all_metrics.extend(metrics)
            except OSError as e:
                print(f"I/O operation failed at iteration {iteration}: {e}")
                all_metrics.append({
                    'iteration': iteration,
                    'error': str(e),
                    'time_since_start': time.time() - start_time
                })
        
        iteration += 1
    
    end_time = time.time()
    
    # Print summary statistics
    print(f"\nTest Summary:")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"Iterations completed: {iteration}")
    
    # Calculate and print I/O statistics
    successful_ops = [m for m in all_metrics if 'error' not in m]
    if successful_ops:
        avg_write_speed = sum(m['write_speed_gbps'] for m in successful_ops) / len(successful_ops)
        avg_read_speed = sum(m['read_speed_gbps'] for m in successful_ops) / len(successful_ops)
        print(f"\nI/O Performance:")
        print(f"Average write speed: {avg_write_speed:.2f} GB/s")
        print(f"Average read speed: {avg_read_speed:.2f} GB/s")
    
    # Print quota hit times
    quota_hits = [m for m in all_metrics if 'error' in m]
    if quota_hits:
        print("\nQuota hits:")
        for hit in quota_hits:
            print(f"Hit at iteration {hit['iteration']}, {hit['time_since_start']:.2f} seconds into test")

if __name__ == "__main__":
    main()
