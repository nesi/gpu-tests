import torch
import time
import argparse
import subprocess
import os
import socket

def run_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    return result.stdout.strip()

def get_gpu_info(gpu_index):
    memory_info = run_command(f"nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i {gpu_index}")
    total_memory, used_memory = map(int, memory_info.split(','))
    temperature = int(run_command(f"nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i {gpu_index}"))
    return total_memory, used_memory, temperature

def gpu_stress_test(device, duration=180):
    print(f"Starting GPU stress test on {device}")
    start_time = time.time()
    
    total_memory = torch.cuda.get_device_properties(device).total_memory
    tensor_size = int((total_memory * 0.9) ** 0.5 // 8)
    
    print(f"Creating tensors of size {tensor_size}x{tensor_size}")
    
    try:
        a = torch.randn(tensor_size, tensor_size, dtype=torch.float64, device=device)
        b = torch.randn(tensor_size, tensor_size, dtype=torch.float64, device=device)
    except torch.cuda.OutOfMemoryError:
        return None, None, "CUDA out of memory error during tensor creation"
    
    peak_memory_usage = 0
    peak_temperature = 0
    
    while time.time() - start_time < duration:
        try:
            c = torch.matmul(a, b)
            d = torch.sin(c) + torch.cos(c)
            result = torch.sum(d)
            
            elapsed = time.time() - start_time
            print(f"Elapsed time: {elapsed:.2f} seconds, Result: {result.item()}")
            
            torch.cuda.synchronize(device)
            
            _, used_memory, temperature = get_gpu_info(device.index)
            peak_memory_usage = max(peak_memory_usage, used_memory)
            peak_temperature = max(peak_temperature, temperature)
        except torch.cuda.CUDAError as e:
            return peak_memory_usage, peak_temperature, f"CUDA error during computation: {str(e)}"
    
    print(f"Test completed on {device}")
    return peak_memory_usage, peak_temperature, None

def main(gpus):
    hostname = socket.gethostname()
    report_file = f"{hostname}.txt"
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
            peak_memory_usage, peak_temperature, cuda_error = gpu_stress_test(device)
            
            if cuda_error:
                status = "FAIL"
                report.append(f"GPU {gpu_index}: {gpu_name}")
                report.append(f"Status: {status}")
                report.append(f"Error: {cuda_error}")
            else:
                memory_utilization = (peak_memory_usage / total_memory) * 100
                status = "PASS" if memory_utilization > 70 and peak_temperature < 85 else "FAIL"
                
                report.append(f"GPU {gpu_index}: {gpu_name}")
                report.append(f"Status: {status}")
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
    parser = argparse.ArgumentParser(description="GPU Hardware Acceptance Test")
    parser.add_argument("--gpu", type=int, nargs='*', help="GPU indices to test (default: all GPUs)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpus = args.gpu if args.gpu is not None else range(torch.cuda.device_count())
        main(gpus)
    else:
        print("No CUDA-capable GPUs found.")