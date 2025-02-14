import torch
import torch.nn as nn
import torch.nn.functional as F

# Define dimensions and rank
input_dim = 512
output_dim = 1024
rank = 8


class TensorTrainLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(TensorTrainLayer, self).__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize core tensors as parameters
        self.core_tensors = nn.ParameterList([
            nn.Parameter(torch.randn(rank, input_dim, output_dim, rank))  # TT decomposition
        ])

    def forward(self, x):
        """Forward pass through the Tensor Train decomposition"""
        x = torch.einsum('bijr,rjk->bik', x, self.core_tensors[0])
        return x


# Example Usage:
# Generate dummy input data (replace with your actual data)
x = torch.randn(64, input_dim, rank)  # Simulating batch input

# Create an instance of the TensorTrainLayer
tt_layer = TensorTrainLayer(input_dim, output_dim, rank)

# Pass the input through the layer
output = tt_layer(x)

# Print the output shape
print(output.shape)