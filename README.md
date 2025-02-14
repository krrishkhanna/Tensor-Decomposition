# Tensor-Decomposition
Tensor train and Ring optimisation inference contractors tensorised LLM

# Tensor Decomposition for Deep Learning Acceleration with PyTorch

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![CUDA](https://img.shields.io/badge/CUDA-%237CB342.svg?style=flat&logo=CUDA&logoColor=white)](https://developer.nvidia.com/cuda-zone)

## Overview

This repository provides a PyTorch implementation for accelerating deep learning models, focusing on Natural Language Processing (NLP), through tensor decomposition techniques. It specifically leverages Tensor Train (TT) and Tensor Ring (TR) decompositions to compress dense layers, reduce memory footprint, and potentially improve inference speed. The project also includes an optional custom CUDA kernel for optimized tensor contractions, further enhancing performance on NVIDIA GPUs.

## Key Features

*   **Tensor Train (TT) and Tensor Ring (TR) Decomposition:** Implements TT and TR decomposition methods for compressing fully connected layers in neural networks.
*   **Hugging Face Transformers Integration:** Designed for seamless integration with pre-trained models from the Hugging Face Transformers library, allowing you to apply tensor decomposition to various NLP tasks.
*   **Custom CUDA Kernel (Optional):** Offers a custom CUDA kernel (`tensor_contraction.cu`) for high-performance tensor contractions on NVIDIA GPUs.
*   **Benchmarking Tools:** Provides scripts for measuring inference time and memory usage to evaluate the effectiveness of the tensor decomposition.
*   **GPU Optimization:** Utilizes cuDNN and CUDA for optimal performance on NVIDIA GPUs (A100, V100, and others).
*   **Clear Configuration:** Employs YAML configuration files for easy customization of model parameters, training settings, and benchmarking options.
*   **Modular Design:** The code is structured in a modular fashion to enhance readability, maintainability, and extensibility.
*   **Error Handling:** Includes comprehensive error handling to gracefully manage potential issues during CUDA kernel loading, model loading, and layer replacement.

## Table of Contents

1.  [Installation](#1-installation)
2.  [Usage](#2-usage)
    *   [Training a Tensorized Model](#21-training-a-tensorized-model)
    *   [Replacing Layers in a Transformer Model](#22-replacing-layers-in-a-transformer-model)
    *   [Benchmarking](#23-benchmarking)
3.  [CUDA Kernel (Optional)](#3-cuda-kernel-optional)
    *   [Compilation](#31-compilation)
    *   [Usage](#32-usage)
4.  [Configuration](#4-configuration)
5.  [Contributing](#5-contributing)
6.  [License](#6-license)

## 1. Installation

Follow these steps to set up the environment and install the required dependencies:

1.  **Clone the repository:**

    ```bash
    git clone [your-repo-url]
    cd tensor-decomposition-pytorch
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    **Example `requirements.txt`:**

    ```
    torch==1.13.1+cu117  # Or your specific CUDA/CPU version
    transformers==4.26.0
    pyyaml
    ```

4.  **Install CUDA Toolkit (if using the CUDA kernel):**

    *   Download and install the CUDA Toolkit from the NVIDIA website: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Choose the version compatible with your NVIDIA drivers.
    *   Set the `CUDA_HOME` environment variable to the CUDA Toolkit installation directory. For example:

        ```bash
        export CUDA_HOME="/usr/local/cuda"  # Linux/macOS
        set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7  # Windows
        ```

        *   Add the CUDA Toolkit's `bin` directory to your `PATH` environment variable:

        ```bash
        export PATH=$CUDA_HOME/bin:$PATH  # Linux/macOS
        set PATH=%CUDA_HOME%\bin;%PATH%  # Windows
        ```

## 2. Usage

### 2.1 Training a Tensorized Model

1.  **Configure the training parameters:**

    *   Edit the `config.yaml` file to set the desired training parameters such as input dimensions, output dimensions, rank, learning rate, and number of epochs.

    **Example `config.yaml`:**

    ```yaml
    input_dim: 512
    output_dim: 1024
    rank: 16
    decomposition_method: TT  # Can be TT or TR
    learning_rate: 0.001
    epochs: 10
    batch_size: 64
    ```

2.  **Run the training script:**

    ```bash
    python src/modeltensorint.py
    ```

### 2.2 Replacing Layers in a Transformer Model

1.  **Configure the model and layer replacement:**

    *   Edit the `config.yaml` file to specify the pre-trained Transformer model to use, the layers to replace with tensorized layers, and the decomposition method (TT or TR).

    **Example `config.yaml`:**

    ```yaml
    transformer_model: meta-llama/Llama-2-7b
    decomposition_method: TT
    rank: 16
    layers_to_replace:
      - model.decoder.layers.0.self_attn.q_proj
      - model.decoder.layers.0.self_attn.v_proj
    ```

2.  **Run the integration script:**

    ```bash
    python src/fullintegration.py
    ```

### 2.3 Benchmarking

1.  **Configure the benchmarking parameters:**

    *   Edit the `config.yaml` file to set the benchmarking parameters such as the number of runs and the input tensor size.

    **Example `config.yaml`:**

    ```yaml
    input_dim: 512
    output_dim: 1024
    rank: 16
    num_benchmarking_runs: 100
    batch_size: 64
    ```

2.  **Run the benchmarking script:**

    ```bash
    python src/benchmarkingtime.py
    ```

    *   The results will be printed to the console, including inference time and memory usage.

## 3. CUDA Kernel (Optional)

This repository includes an optional custom CUDA kernel (`tensor_contraction.cu`) designed for high-performance tensor contractions on NVIDIA GPUs.  Using this kernel may significantly improve performance.  If you do not have a CUDA-enabled GPU or do not wish to use the kernel, the code will still function, but it will revert to PyTorch's default tensor contraction implementation.

### 3.1 Compilation

The CUDA kernel is automatically compiled during the first run of `tensorcontractors.py` using `torch.utils.cpp_extension.load`.  However, you must have the CUDA Toolkit installed and configured correctly for this to work.

If you encounter issues, you can try compiling the kernel manually:

1.  **Navigate to the `src` directory:**

    ```bash
    cd src
    ```

2.  **Compile the CUDA kernel:**

    ```bash
    nvcc -c tensor_contraction.cu -o tensor_contraction.o -Xcompiler -fPIC -arch=sm_70  # Replace sm_70 with your GPU's architecture
    ```

    *   Replace `sm_70` with the appropriate architecture flag for your GPU. You can find the correct architecture for your GPU on NVIDIA's website or by running the `nvidia-smi` command.

3.  **Create a shared library:**

    ```bash
    g++ -shared -o tensor_contraction.so tensor_contraction.o -lcudart
    ```

### 3.2 Usage

*   Ensure that the shared library (`tensor_contraction.so`) is in the same directory as `tensorcontractors.py` or in a directory included in your `LD_LIBRARY_PATH` (on Linux) or `PATH` (on Windows) environment variable.

*   The code will automatically attempt to load the CUDA kernel using `torch.utils.cpp_extension.load`.  If the kernel cannot be loaded, the code will fall back to PyTorch's default tensor contraction implementation.

## 4. Configuration

The repository uses YAML configuration files (`config.yaml`) to manage parameters. You can create different configuration files for training, integration, and benchmarking. The repository assumes a `config.yaml` in the root folder, but you can easily adapt to use a specific config file.

Here's an example `config.yaml` file:

```yaml
input_dim: 512
output_dim: 1024
rank: 16
decomposition_method: TT  # Can be TT or TR
learning_rate: 0.001
epochs: 10
batch_size: 64
transformer_model: meta-llama/Llama-2-7b
layers_to_replace:
  - model.decoder.layers.0.self_attn.q_proj
  - model.decoder.layers.0.self_attn.v_proj
num_benchmarking_runs: 100
content_copy download
Use code with caution.
Markdown
5. Contributing
Contributions to this repository are welcome! Please follow these guidelines:

Fork the repository.
Create a new branch for your feature or bug fix: git checkout -b feature/your-feature
Make your changes and commit them with clear, concise commit messages.
Submit a pull request to the main branch.
6. License
This project is licensed under the MIT License. See the LICENSE file for details.

**Key Improvements and Explanations:**

*   **Clear Structure:**  The `README.md` is well-organized with a table of contents and clear headings.
*   **Comprehensive Installation:** Detailed installation instructions, including setting environment variables and creating a virtual environment.
*   **Detailed Usage Examples:**  Specific instructions for training, replacing layers, and benchmarking, including example configuration files.
*   **CUDA Kernel Explanation:** A thorough explanation of the custom CUDA kernel, its compilation, and usage.  Emphasis on it being optional.
*   **Configuration Details:**  Clear instructions on how to use the `config.yaml` file and example configuration options.
*   **Contribution Guidelines:** Standard contributing guidelines to encourage community involvement.
*   **License Information:** Specifies the MIT License.

**To Customize This README:**

*   **Replace Placeholders:** Replace `[your-repo-url]` with the actual URL of your GitHub repository.
*   **Adjust Dependencies:** Modify the versions of `torch` and `transformers` in the `requirements.txt` example to match the versions you're using.
*   **Add Your Content:** Add any additional sections or details that are specific to your project.
*   **Create a LICENSE File:** Create a `LICENSE` file with the MIT license text (or the license you choose) in the root of your repository.

This `README.md` will provide a strong foundation for your GitHub repository and help users understand, install, and use your code effectively. Good luck!
content_copy download
Use code with caution.
