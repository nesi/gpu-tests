import torch
import time
import subprocess
import threading
import os

def run_nvidia_smi_topo(log_file, interval=60, stop_event=None):
    while not stop_event.is_set():
        try:
            result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a') as f:
                f.write(f"\n\n--- Topology at {timestamp} ---\n")
                f.write(result.stdout)
            time.sleep(interval)
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")
            break

def test_nvlink_transfer(src_gpu, dst_gpu, tensor_size_gb=1):
    tensor_size = int(tensor_size_gb * 1024 * 1024 * 1024 / 4)
    src_tensor = torch.rand(tensor_size, dtype=torch.float32, device=f'cuda:{src_gpu}')
    torch.cuda.synchronize(src_gpu)
    start_time = time.time()
    dst_tensor = src_tensor.to(f'cuda:{dst_gpu}')
    torch.cuda.synchronize(dst_gpu)
    end_time = time.time()
    transfer_time = end_time - start_time
    transfer_speed_gbps = (tensor_size_gb * 8) / transfer_time
    return transfer_speed_gbps

def test_all_gpu_pairs(num_gpus, tensor_size_gb=1):
    results = {}
    for src_gpu in range(num_gpus):
        for dst_gpu in range(num_gpus):
            if src_gpu != dst_gpu:
                print(f"Testing transfer from GPU {src_gpu} to GPU {dst_gpu}...")
                speed = test_nvlink_transfer(src_gpu, dst_gpu, tensor_size_gb)
                results[(src_gpu, dst_gpu)] = speed
                print(f"Transfer speed: {speed:.2f} Gb/s")
    return results

def main():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs detected: {num_gpus}")

    if num_gpus < 2:
        print("At least 2 GPUs are required to test NVLink connections.")
        return

    # Set up topology logging
    log_file = "nvidia_smi_topo_log.txt"
    stop_event = threading.Event()
    topo_thread = threading.Thread(target=run_nvidia_smi_topo, args=(log_file, 60, stop_event))
    topo_thread.start()

    try:
        # Test NVLink connections
        results = test_all_gpu_pairs(num_gpus)

        # Print summary
        print("\nNVLink Test Results Summary:")
        for (src, dst), speed in results.items():
            print(f"GPU {src} to GPU {dst}: {speed:.2f} Gb/s")

        # Check for potential issues
        for (src, dst), speed in results.items():
            if speed < 20:
                print(f"\nWarning: Low transfer speed detected between GPU {src} and GPU {dst}.")
                print("This might indicate a problem with the NVLink connection or configuration.")

    finally:
        # Stop the topology logging thread
        stop_event.set()
        topo_thread.join()

    print(f"\nTopology log has been saved to {os.path.abspath(log_file)}")

if __name__ == "__main__":
    main()
