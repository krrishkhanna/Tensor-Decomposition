from torch.utils.cpp_extension import load
import torch.nn as nn  # Import nn

# Custom CUDA kernel for efficient tensor contraction
try:
    tensor_contraction_cuda = load(
        "tensor_contraction", sources=["tensor_contraction.cu"], verbose=True
    )
except Exception as e:
    print(f"Error loading CUDA kernel: {e}")
    tensor_contraction_cuda = None  # Handle the case where loading fails!


class FastTensorContractor(nn.Module):
    def forward(self, a, b):
        if tensor_contraction_cuda is None:
            raise RuntimeError("CUDA kernel not loaded.")  # Prevent crash

        return tensor_contraction_cuda.contract(a, b)