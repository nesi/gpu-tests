## `torch_stress_test.py`

The test will run on all available GPUs sequentially

1. checks for available CUDA-capable GPUs.
2. For each GPU, it runs a stress test for at least 3 minutes (180 seconds) as per `duration=180`.
3. The test performs continuous matrix multiplications and element-wise operations on large tensors.
4. prints progress updates, including the elapsed time and a result from the computations.

## `full_memory_torch_stress_test.py`

On top of the following changes to utilise more GPU memory during the test, this allows us to choose the GPU of interest ( 0 , 1, etc) as it can be executed with `python full_memory_torch_stress_test.py --gpu 0`   ,etc. 

1. Calculate the maximum tensor size that will fit in GPU memory, using about 90% of the available memory. This leaves some headroom to avoid out-of-memory errors.
2. use `torch.float64` (64-bit floating point) to increase memory usage and computational intensity.
3. reports the total memory of the GPU being tested.
4. `torch.cuda.synchronize(device)` to ensure each operation completes before moving to the next iteration.


## `report_full_memory_torch_stress_test.py`

This updated script incorporates all the requested features:

1. `Multi-GPU Support`: If no `--gpu` argument is provided, it tests all available GPUs.
2. `nvidia-smi` Integration: The script uses nvidia-smi to gather memory utilization and temperature data.
3. Report Generation: It creates a text file named after the hostname (e.g., hostname.txt) with a report for each GPU tested.
4. `PASS/FAIL` Criteria: The test is considered a PASS if memory utilization exceeds 85% and the peak temperature stays below 85°C. You can adjust these thresholds as needed.