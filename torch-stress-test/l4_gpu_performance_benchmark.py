import torch
import time
import argparse
import subprocess
import socket

def run_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    return result.stdout.strip()

def get_gpu_info(gpu_index):
    memory_info = run_command(f"nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i {gpu_index}")
    total_memory, used_memory = map(int, memory_info.split(','))
    temperature = int(run_command(f"nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i {gpu_index}"))
    return total_memory, used_memory, temperature

def gpu_benchmark(device, duration=600):  # 10 minutes
    print(f"Starting GPU benchmark on {device}")
    start_time = time.time()
    
    # Get available GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    _, used_memory_before, _ = get_gpu_info(device.index)
    
    # FIXED: Proper tensor size calculation for FP32
    # Start with a conservative fixed size and adjust if needed
    tensor_size = 8000  # This should work on most L4 GPUs (24GB)
    
    # Enable TF32 for Tensor Core utilization on Ampere and newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Creating tensors of size {tensor_size}x{tensor_size}")
    
    try:
        a = torch.randn(tensor_size, tensor_size, dtype=torch.float32, device=device)
        b = torch.randn(tensor_size, tensor_size, dtype=torch.float32, device=device)
    except torch.cuda.OutOfMemoryError:
        # If we can't allocate with the initial size, try with a smaller size
        print("Out of memory with initial tensor size, reducing...")
        tensor_size = 6000
        print(f"Trying with reduced tensor size: {tensor_size}x{tensor_size}")
        try:
            a = torch.randn(tensor_size, tensor_size, dtype=torch.float32, device=device)
            b = torch.randn(tensor_size, tensor_size, dtype=torch.float32, device=device)
        except torch.cuda.OutOfMemoryError:
            # If still OOM, try one more time with an even smaller size
            tensor_size = 4000
            print(f"Trying with further reduced tensor size: {tensor_size}x{tensor_size}")
            try:
                a = torch.randn(tensor_size, tensor_size, dtype=torch.float32, device=device)
                b = torch.randn(tensor_size, tensor_size, dtype=torch.float32, device=device)
            except torch.cuda.OutOfMemoryError:
                return None, None, None, "CUDA out of memory error during tensor creation"
    
    peak_memory_usage = 0
    peak_temperature = 0
    total_flops = 0
    iteration_count = 0
    
    try:
        while time.time() - start_time < duration:
            iteration_start = time.time()
            
            c = torch.matmul(a, b)
            d = torch.sin(c) + torch.cos(c)
            result = torch.sum(d)
            
            torch.cuda.synchronize(device)
            
            iteration_end = time.time()
            iteration_time = iteration_end - iteration_start
            
            # Calculate FLOPS for this iteration
            # Matrix multiplication: 2 * N^3 FLOPs
            # Sin and Cos: 2 * N^2 FLOPs each
            # Sum: N^2 FLOPs
            flops_per_iteration = 2 * tensor_size**3 + 5 * tensor_size**2
            total_flops += flops_per_iteration
            iteration_count += 1
            
            _, used_memory, temperature = get_gpu_info(device.index)
            peak_memory_usage = max(peak_memory_usage, used_memory)
            peak_temperature = max(peak_temperature, temperature)
            
            print(f"Iteration {iteration_count}: Time: {iteration_time:.2f}s, FLOPS: {flops_per_iteration/iteration_time:.2e}, Memory: {used_memory}MB, Temp: {temperature}°C")
            
            if temperature >= 80:
                print("Warning: Approaching thermal limits - consider cooling improvements")
    
    except torch.cuda.CUDAError as e:
        return peak_memory_usage, peak_temperature, None, f"CUDA error during computation: {str(e)}"
    
    total_time = time.time() - start_time
    average_flops = total_flops / total_time
    
    print(f"Benchmark completed on {device}")
    return peak_memory_usage, peak_temperature, average_flops, None

def main(gpus):
    hostname = socket.gethostname()
    report_file = f"{hostname}_performance_fp32.txt"
    report = []

    for gpu_index in gpus:
        if gpu_index >= torch.cuda.device_count():
            print(f"Error: GPU index {gpu_index} is out of range. Available GPUs: {torch.cuda.device_count()}")
            continue

        device = torch.device(f'cuda:{gpu_index}')
        gpu_name = torch.cuda.get_device_name(gpu_index)
        total_memory, _, _ = get_gpu_info(gpu_index)
        print(f"Testing GPU {gpu_index}: {gpu_name} (Total memory: {total_memory} MB)")
        
        try:
            peak_memory_usage, peak_temperature, average_flops, error = gpu_benchmark(device)
            
            if error:
                status = "FAIL"
                report.append(f"GPU {gpu_index}: {gpu_name}")
                report.append(f"Status: {status}")
                report.append(f"Error: {error}")
            else:
                memory_utilization = (peak_memory_usage / total_memory) * 100
                # L4 specific pass criteria
                status = "PASS" if (
                    memory_utilization > 70 and 
                    peak_temperature < 80 and 
                    average_flops > 60e12  # 60 TFLOPS threshold for L4
                ) else "FAIL"
                
                report.append(f"GPU {gpu_index}: {gpu_name}")
                report.append(f"Status: {status}")
                report.append(f"Performance: {average_flops:.2e} FLOPS")
                report.append(f"Peak Memory Utilization: {memory_utilization:.2f}%")
                report.append(f"Peak Temperature: {peak_temperature}°C")
            
            report.append("")
        except Exception as e:
            report.append(f"GPU {gpu_index}: {gpu_name}")
            report.append(f"Status: FAIL")
            report.append(f"Error: Unexpected error - {str(e)}")
            report.append("")

    with open(report_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Performance Benchmark for FP32 (optimized for NVIDIA L4)")
    parser.add_argument("--gpu", type=int, nargs='*', help="GPU indices to test (default: all GPUs)")
    parser.add_argument("--duration", type=int, default=600, help="Duration of the benchmark in seconds (default: 600)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpus = args.gpu if args.gpu is not None else range(torch.cuda.device_count())
        main(gpus)
    else:
        print("No CUDA-capable GPUs found.")
