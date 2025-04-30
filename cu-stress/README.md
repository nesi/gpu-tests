

## How to Adjust Memory Usage



```c
float memoryUsagePercentage = 0.5; // Using 50% of available memory
```
- For light testing: `memoryUsagePercentage = 0.1` (10% of GPU memory)
- For medium testing: `memoryUsagePercentage = 0.5` (50% of GPU memory)
- For heavy testing: `memoryUsagePercentage = 0.8` (80% of GPU memory)
- For maximum stress: `memoryUsagePercentage = 0.9` (90% of GPU memory)


### How Memory Allocation Works
The code allocates three large arrays (`d_data1`, `d_data2`, and `d_data3`), which is why the total requested memory is divided by 3. 
Each array gets approximately 1/3 of the requested memory, and the stress test performs operations between these arrays.
