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

2. GPU Selection:
    - Each worker process is assigned to a specific GPU using `torch.device(f'cuda:{gpu_id}')`.

3. Memory Usage:
    - Each GPU will use approximately 72GB of memory, so you'll need 4 A100 GPUs with at least 80GB of memory each.

4. I/O Operations:
    - I/O operations are now performed independently for each GPU process.
    - Each process creates and operates on its own file to avoid I/O conflicts.

5. Run Time:
    - Each GPU will run for at least 5 minutes, as before.