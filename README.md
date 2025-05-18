# CUDA-parallelSSSP

This project implements the **Single-Source Shortest Path (SSSP)** algorithm using CUDA, with both static and dynamic settings for efficient parallel computation on GPUs. It demonstrates the power of CUDA for accelerating graph algorithms.

---

## 🚀 Features

* **Multiple Implementations**:

  * CPU (Sequential, Optimized)
  * GPU (Basic Parallel, Optimized)
* **Static and Dynamic Settings**:

  * Static Setting: Fixed settings for optimized performance.
  * Dynamic Setting: Customizable settings for experimentation.
* **Flag Optimization**:

  * Specialized GPU implementations with flag-based optimizations for faster convergence.

---

## 📁 Project Structure

```
CUDA-parallelSSSP/
├── dynamicSetting/
│   └── gpuImplementations/
│       ├── sssp_parallel.cu
│       └── src.cu
├── staticSetting/
│   ├── cpuImplementations/
│   │   ├── sssp_sequential.cu
│   │   ├── sssp_sequential_flags.cu
│   │   └── sssp_sequential_flags_optimized.cu
│   └── gpuImplementations/
│       ├── sssp_parallel.cu
│       ├── sssp_parallel_flags.cu
│       ├── sssp_parallel_flags_optimized_version1.cu
│       ├── sssp_parallel_flags_optimized_version2.cu
│       └── sssp_parallel_flags_optimized_version3.cu
├── LICENSE
└── README.md
```

* **dynamicSetting/**: Contains GPU implementations with flexible configurations.
* **staticSetting/**: Contains both CPU and GPU implementations with predefined settings.

---

## 🛠️ Building the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/CodeCraftsmanSandeep/CUDA-parallelSSSP.git
   cd CUDA-parallelSSSP
   ```
