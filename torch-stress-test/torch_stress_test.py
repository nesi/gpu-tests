import torch
import time

def gpu_stress_test(device, duration=180):
    print(f"Starting GPU stress test on {device}")
    start_time = time.time()
    
    # Create a large tensor
    size = 10000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
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
    
    print(f"Test completed on {device}")

def main():
    # Test on all available GPUs
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        gpu_name = torch.cuda.get_device_name(i)
        print(f"Testing GPU {i}: {gpu_name}")
        gpu_stress_test(device)
        print("\n")

if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("No CUDA-capable GPUs found.")
