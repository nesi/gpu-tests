import torch
import time

def test_nvlink_transfer(src_gpu, dst_gpu, tensor_size_gb=1):
    # Create a large tensor on the source GPU
    tensor_size = int(tensor_size_gb * 1024 * 1024 * 1024 / 4)  # size in number of float32 elements
    src_tensor = torch.rand(tensor_size, dtype=torch.float32, device=f'cuda:{src_gpu}')

    # Ensure the tensor is on the GPU
    torch.cuda.synchronize(src_gpu)

    # Perform the transfer and measure time
    start_time = time.time()
    dst_tensor = src_tensor.to(f'cuda:{dst_gpu}')
    torch.cuda.synchronize(dst_gpu)
    end_time = time.time()

    # Calculate transfer speed
    transfer_time = end_time - start_time
    transfer_speed_gbps = (tensor_size_gb * 8) / transfer_time  # Convert GB/s to Gb/s

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
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs detected: {num_gpus}")

    if num_gpus < 2:
        print("At least 2 GPUs are required to test NVLink connections.")
        return

    # Test NVLink connections
    results = test_all_gpu_pairs(num_gpus)

    # Print summary
    print("\nNVLink Test Results Summary:")
    for (src, dst), speed in results.items():
        print(f"GPU {src} to GPU {dst}: {speed:.2f} Gb/s")

    # Check for potential issues
    for (src, dst), speed in results.items():
        if speed < 20:  # This threshold might need adjustment based on your specific hardware
            print(f"\nWarning: Low transfer speed detected between GPU {src} and GPU {dst}.")
            print("This might indicate a problem with the NVLink connection or configuration.")

if __name__ == "__main__":
    main()
