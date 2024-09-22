1. GPU computation to run for at least 5 minutes
2. Utilize 70-75GB of GPU memory
3. Include a heavy I/O component

### `single-gpu.py`:

1. GPU Computation and Memory Usage:
    - Creates a large tensor of approximately 72GB on the GPU.
    - Performs continuous matrix multiplications, sine, and exponential operations on this tensor.

2. Run Time:

    - The main loop continues until at least 5 minutes have passed.

3. I/O Operations:

    - Every 10 iterations of GPU computation, it performs I/O operations.
    - Writes and reads a 10GB file 50 times per I/O operation cycle.

### `multi-gpu.py`

1. Multi-GPU Support:
    - The script now uses Python's `multiprocessing` module to create a separate process for each GPU.
    - Each process runs the `gpu_worker` function, which performs computations on a specific GPU.
    - Make sure you have the specified number of GPUs available (default is 4, but you can change the `num_gpus` parameter in the main function)

2. GPU Selection:
    - Each worker process is assigned to a specific GPU using `torch.device(f'cuda:{gpu_id}')`.

3. Memory Usage:
    - Each GPU will use approximately 72GB of memory, so you'll need 4 A100 GPUs with at least 80GB of memory each.

4. I/O Operations:
    - I/O operations are now performed independently for each GPU process.
    - Each process creates and operates on its own file to avoid I/O conflicts.

5. Run Time:
    - Each GPU will run for at least 5 minutes, as before.


### `nvlink_test.py`

What does this do ?

* Detects the number of available GPUs. > Tests data transfer between all pairs of GPUs. > Measures the transfer speed for each pair. > Prints a summary of the results. > Warns about potentially low transfer speeds.

**Notes**

Some important notes:

1. This test assumes that  GPUs are connected via NVLink. If they're not, you'll still see transfer speeds, but they'll be lower than what NVLink can provide.
2. The script uses a 1GB tensor for transfer by default. This can be adjusted by changing the `tensor_size_gb` parameter in the `test_all_gpu_pairs` function call if you want to test with larger data transfers.
3. The warning threshold (20 Gb/s) is a conservative estimate. Actual NVLink speeds can be much higher (up to 300 Gb/s for NVLink 3.0).
4. This test measures peer-to-peer GPU transfer speeds, which should use NVLink if it's available and properly configured. However, it doesn't directly verify that NVLink is being used. To confirm NVLink usage, you can use NVIDIA's `nvidia-smi topo -m` command outside of this script.

**Recommendation**

For a more comprehensive test, you might want to run this multiple times and average the results, as there can be some variation in transfer speeds.