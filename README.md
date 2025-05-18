# CUDA-parallelSSSP

This project implements a **Single-Source Shortest Path (SSSP)** algorithm using **CUDA** to parallelize the process and accelerate performance. The goal is to efficiently compute the shortest paths from a source node to all other nodes in a graph, utilizing the power of GPUs to handle large-scale graph data.

## Features

- **Parallelization**: The algorithm is parallelized using CUDA, enabling faster processing of large graphs.
- **Optimized Data Structures**: Efficient data structures are used to represent the graph and store distances.
- **Graph Representation**: The graph is represented as an adjacency list, allowing for efficient traversal.
- **CUDA Kernels**: Custom CUDA kernels are implemented for key steps in the algorithm.

## Requirements

- **CUDA**: Ensure you have a CUDA-capable GPU and the CUDA toolkit installed.
- **C++ Compiler**: A C++ compiler compatible with CUDA.
- **Make**: Used for building the project.
- **Linux/MacOS**: The project is primarily tested on Linux and MacOS environments.

### System Requirements

- CUDA version: 7.0 or later
- GPU with at least 1 GB of memory (recommended)
- GCC version 7 or higher (for Linux) or Clang (for MacOS)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/CUDA-parallelSSSP.git
