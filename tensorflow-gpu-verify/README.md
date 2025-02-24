## `tf_gpu_verify.py`:

**note** set CUDA installation path before running: `export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/your/cuda"`

* Verify TensorFlow can see the GPU
* Create a moderately large dataset and neural network
* Train for approximately 60 seconds
* Print progress information

### sample output `gpu_metrics.log`
GPU metrics logging at several points:

- Before training starts
- During training (after each epoch)
- After training completes
