import torch
import torch.nn as nn
import torch.nn.functional as F

# Define dimensions and rank (or import them)
input_dim = 512
output_dim = 1024
rank = 8


class TensorRingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(TensorRingLayer, self).__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize core tensors as parameters
        self.core_tensors = nn.ParameterList([
            nn.Parameter(torch.randn(rank, input_dim, rank)),  # First core
            nn.Parameter(torch.randn(rank, output_dim, rank))  # Second core
        ])

    def forward(self, x):
        """Forward pass through the Tensor Ring decomposition"""
        x = torch.einsum('bir,rjk->bik', x, self.core_tensors[0])
        x = torch.einsum('bik,rko->bio', x, self.core_tensors[1])
        return x


# Example Usage:
# Generate dummy input data (replace with your actual data)
x = torch.randn(64, input_dim, rank)

# Create an instance of the TensorRingLayer
tr_layer = TensorRingLayer(input_dim, output_dim, rank)

# Pass the input through the layer
output = tr_layer(x)

# Print the output shape
print(output.shape)