import torch
import time
import argparse

def gpu_stress_test(device, duration=180):
    print(f"Starting GPU stress test on {device}")
    start_time = time.time()
    
    # Get the total amount of memory on the GPU
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # Calculate the size of the tensor to use most of the available memory
    # We'll use 90% of the available memory to leave some headroom
    tensor_size = int((total_memory * 0.9) ** 0.5 // 8)  # 8 bytes per float64
    
    print(f"Creating tensors of size {tensor_size}x{tensor_size}")
    
    # Create large tensors that fill most of the GPU memory
    a = torch.randn(tensor_size, tensor_size, dtype=torch.float64, device=device)
    b = torch.randn(tensor_size, tensor_size, dtype=torch.float64, device=device)
    
    while time.time() - start_time < duration:
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        # Perform element-wise operations
        d = torch.sin(c) + torch.cos(c)
        # Reduce
        result = torch.sum(d)
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds, Result: {result.item()}")
        
        # Force synchronization to ensure operation completion
        torch.cuda.synchronize(device)
    
    print(f"Test completed on {device}")

def main(gpu_index):
    if gpu_index >= torch.cuda.device_count():
        print(f"Error: GPU index {gpu_index} is out of range. Available GPUs: {torch.cuda.device_count()}")
        return

    device = torch.device(f'cuda:{gpu_index}')
    gpu_name = torch.cuda.get_device_name(gpu_index)
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"Testing GPU {gpu_index}: {gpu_name} (Total memory: {total_memory_gb:.2f} GB)")
    
    gpu_stress_test(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Hardware Acceptance Test")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to test (default: 0)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        main(args.gpu)
    else:
        print("No CUDA-capable GPUs found.")
