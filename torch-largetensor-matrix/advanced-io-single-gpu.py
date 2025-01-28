#!/usr/bin/env python3

import torch
import time
import os
from datetime import datetime

def get_timestamp():
    """
    Returns a formatted timestamp for logging purposes.
    Helps track exact timing of events during the test.
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def create_large_tensor(size_gb):
    """Creates a large tensor on GPU to simulate memory usage."""
    num_elements = int(size_gb * 1024 * 1024 * 1024 / 4)
    return torch.rand(num_elements, device='cuda')

def perform_computations(tensor, iterations):
    """
    Performs GPU computations and tracks completion time of each cycle.
    Returns both the processed tensor and timing information.
    """
    computation_times = []
    chunk_size = 1000000
    
    for iter_num in range(iterations):
        cycle_start = time.time()
        for i in range(0, tensor.size(0), chunk_size):
            chunk = tensor[i:i+chunk_size]
            chunk = torch.sin(chunk)
            chunk = torch.exp(chunk)
            tensor[i:i+chunk_size] = chunk
        
        computation_times.append({
            'iteration': iter_num,
            'duration': time.time() - cycle_start,
            'timestamp': get_timestamp()
        })
    
    return tensor, computation_times

def io_operations(file_size_gb, num_operations):
    """
    Performs I/O operations with detailed tracking of successful writes.
    Now includes partial write tracking before quota errors.
    """
    filename = "test_file.bin"
    metrics = []
    chunk_size = 128 * 1024 * 1024  # 128MB chunks for better tracking
    total_size = int(file_size_gb * 1024 * 1024 * 1024)
    
    for op in range(num_operations):
        start_time = time.time()
        bytes_written = 0
        
        try:
            # Write data in chunks to track partial success
            with open(filename, 'wb') as f:
                while bytes_written < total_size:
                    chunk_data = os.urandom(min(chunk_size, total_size - bytes_written))
                    f.write(chunk_data)
                    bytes_written += len(chunk_data)
                    f.flush()
                    
                    # Add small delay between chunks to potentially avoid quota
                    time.sleep(0.1)
            
            # If write succeeds, attempt read
            read_start = time.time()
            with open(filename, 'rb') as f:
                _ = f.read()
            read_end = time.time()
            
            metrics.append({
                'operation': op,
                'write_success': True,
                'bytes_written': bytes_written,
                'write_duration': time.time() - start_time,
                'read_duration': read_end - read_start,
                'timestamp': get_timestamp()
            })
            
        except OSError as e:
            metrics.append({
                'operation': op,
                'write_success': False,
                'bytes_written': bytes_written,
                'error': str(e),
                'timestamp': get_timestamp()
            })
            break
        
        finally:
            try:
                os.remove(filename)
            except OSError:
                pass
            
            # Increased delay between operations to find "safe" rate
            time.sleep(2.0)  # Adjustable delay between operations
    
    return metrics

def main():
    # Adjusted parameters for more controlled I/O
    gpu_memory_usage_gb = 72
    run_time_minutes = 5
    io_file_size_gb = 10
    io_operations_count = 50
    io_test_frequency = 20  # Increased from 10 to reduce I/O frequency
    
    print(f"Starting A100 GPU test at {get_timestamp()}")
    
    large_tensor = create_large_tensor(gpu_memory_usage_gb)
    
    start_time = time.time()
    iteration = 0
    all_metrics = {
        'io_operations': [],
        'gpu_computations': [],
        'quota_hits': []
    }
    
    while time.time() - start_time < run_time_minutes * 60:
        # GPU computations with timing
        tensor, comp_times = perform_computations(large_tensor, 10)
        all_metrics['gpu_computations'].extend(comp_times)
        
        # I/O operations at reduced frequency
        if iteration % io_test_frequency == 0:
            print(f"\nStarting I/O operations at iteration {iteration} ({get_timestamp()})")
            try:
                metrics = io_operations(io_file_size_gb, io_operations_count)
                all_metrics['io_operations'].extend(metrics)
                
                # Check for quota hits in metrics
                for m in metrics:
                    if not m.get('write_success', True):
                        all_metrics['quota_hits'].append({
                            'iteration': iteration,
                            'time_since_start': time.time() - start_time,
                            'bytes_written': m['bytes_written'],
                            'timestamp': m['timestamp']
                        })
                        print(f"Quota hit after writing {m['bytes_written']/1024/1024:.2f}MB")
            
            except Exception as e:
                print(f"Error during I/O operations: {e}")
        
        iteration += 1
    
    end_time = time.time()
    
    # Comprehensive results reporting
    print(f"\nTest Summary (completed at {get_timestamp()}):")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"Iterations completed: {iteration}")
    
    print("\nGPU Computation Statistics:")
    comp_times = [t['duration'] for t in all_metrics['gpu_computations']]
    print(f"Average computation cycle time: {sum(comp_times)/len(comp_times):.3f} seconds")
    
    print("\nI/O Operation Statistics:")
    successful_writes = [m for m in all_metrics['io_operations'] if m.get('write_success', False)]
    if successful_writes:
        print(f"Successful complete writes: {len(successful_writes)}")
        avg_write_time = sum(m['write_duration'] for m in successful_writes) / len(successful_writes)
        print(f"Average successful write time: {avg_write_time:.2f} seconds")
    
    print("\nQuota Hits:")
    for hit in all_metrics['quota_hits']:
        print(f"Hit at iteration {hit['iteration']}, "
              f"{hit['time_since_start']:.2f} seconds into test, "
              f"after writing {hit['bytes_written']/1024/1024:.2f}MB")

if __name__ == "__main__":
    main()
