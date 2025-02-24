#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import subprocess
from datetime import datetime

def get_gpu_usage():
    """Capture nvidia-smi output"""
    try:
        nvidia_smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits']
        ).decode('utf-8').strip()
        return nvidia_smi
    except subprocess.CalledProcessError:
        return "Error: Unable to fetch GPU metrics"

def log_gpu_metrics(log_file=None):
    """Log GPU metrics to file or stdout"""
    gpu_info = get_gpu_usage()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] GPU Metrics:\n{gpu_info}\n"
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_message)
    print(log_message)

def verify_gpu(log_file=None):
    print("TensorFlow version:", tf.__version__)
    
    # Check if GPU is available
    print("\nGPU Devices:", tf.config.list_physical_devices('GPU'))
    
    # Initial GPU state
    print("\nInitial GPU State:")
    log_gpu_metrics(log_file)
    
    # Create a large dataset to ensure GPU utilization
    print("\nGenerating test data...")
    size = 100000
    features = 1000
    X = np.random.randn(size, features).astype(np.float32)
    y = np.random.randint(0, 2, size).astype(np.float32)
    
    # Create a model that's large enough to benefit from GPU
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(features,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    print("\nStarting training...")
    start_time = time.time()
    
    # Train for approximately 1 minute
    while (time.time() - start_time) < 60:
        model.fit(X, y,
                 epochs=1,
                 batch_size=256,
                 verbose=1)
        
        current_time = time.time() - start_time
        print(f"\nElapsed time: {current_time:.2f} seconds")
        
        # Log GPU metrics during training
        log_gpu_metrics(log_file)
    
    print("\nTraining completed!")
    
    # Final GPU state
    print("\nFinal GPU State:")
    log_gpu_metrics(log_file)

if __name__ == "__main__":
    # You can specify a log file or leave it as None to print to stdout only
    verify_gpu(log_file="gpu_metrics.log")
