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
4. `PASS/FAIL` Criteria:
  - `PASS`: Memory utilization > 85% AND peak temperature < 85°C AND no CUDA errors
  - `FAIL`: Memory utilization ≤ 85% OR peak temperature ≥ 85°C OR any CUDA error occurred


## Note on the outputs

Runnning the full memory tests will create an output similar to below for each GPU

```bash
Creating tensors of size 18252x18252
Elapsed time: 29.84 seconds, Result: 30158.399930649088
Elapsed time: 29.95 seconds, Result: 30158.399930649088
Elapsed time: 59.66 seconds, Result: 30158.399930649088
```

The Result value(e.g., -6771.25387119431) is the sum of all elements in the tensor d. Let's break down the calculation step by step:

1. `c = torch.matmul(a, b):`
This performs a matrix multiplication of tensors a and b. The result `c` is a new tensor of the same size as a and b.
2. `d = torch.sin(c) + torch.cos(c)`:
This applies the sine function to each element of `c`, then adds it to the cosine of each element of `c`. The result `d` is a tensor of the same size as `c`.
3. `result = torch.sum(d):`
This sums up all the elements in tensor d into a single scalar value.
4. `result.item()`:
This extracts the scalar value from the PyTorch tensor and converts it to a Python number.

The specific value `-6771.25387119431` doesn't have any particular significance in terms of GPU performance or health. It's just the mathematical result of the operations performed.
Here's why this value is used in the test:

*  **Forcing Computation**: By performing these calculations and using the result, we ensure that the GPU actually computes something and doesn't optimize away our operations.
* **Consistency Check**: If the GPU is functioning correctly, this value should be consistent across iterations (as you can see in your output where the value is the same for both lines).
* **Preventing Optimizations** : Summing to a single value and printing it prevents the compiler from optimizing away the entire computation as unused.