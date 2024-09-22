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


### `nvlink_test_with_topo.py`

1. `run_nvidia_smi_topo` function
    - This function runs `nvidia-smi topo -m` at regular intervals (default is every 60 seconds).
    -  The output is timestamped and appended to a log file, *nvidia_smi_topo_log.txt*


2. Threading:
    - We use a separate thread to run the `nvidia-smi topo -m` command periodically.
    - This allows the topology logging to occur concurrently with the NVLink transfer tests.


3. Cleanup: try/finally block to ensure that the topology logging thread is properly stopped, even if an error occurs during the NVLink tests.


**Sample Output**

```bash
--- Topology at 2024-09-23 09:34:37 ---
	GPU0	GPU1	GPU2	GPU3	NIC0	CPU Affinity	NUMA Affinity
GPU0	 X 	NV4	NV4	NV4	SYS	0,64	0-3
GPU1	NV4	 X 	NV4	NV4	SYS	0,64	0-3
GPU2	NV4	NV4	 X 	NV4	SYS	0,64	0-3
GPU3	NV4	NV4	NV4	 X 	PHB	0,64	0-3
NIC0	SYS	SYS	SYS	PHB	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0

```
### `gpuwarmup_nvlink_test_with_topo.py`



1. `warm_up_gpus` function:

    - This function performs small transfers between all GPU pairs before the actual test.
    - It uses a small tensor (1024 elements) to quickly initialize CUDA contexts and warm up the connections.


2. Modified the main function:

    - Added a call to `warm_up_gpus(gpu_order)` after shuffling the GPU order but before starting the actual tests.


3. Kept the shuffling mechanism:

    - This ensures that we still get a randomized order of GPU testing, which helps verify that the issue isn't tied to a specific GPU