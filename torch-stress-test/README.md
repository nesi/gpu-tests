## `torch_stress_test.py`

The test will run on all available GPUs sequentially

1. checks for available CUDA-capable GPUs.
2. For each GPU, it runs a stress test for at least 3 minutes (180 seconds) as per `duration=180`.
3. The test performs continuous matrix multiplications and element-wise operations on large tensors.
4. prints progress updates, including the elapsed time and a result from the computations.

## `full_memory_torch_stress_test.py`

