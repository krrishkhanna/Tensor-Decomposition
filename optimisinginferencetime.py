import torch
import torch.nn as nn

# Import the Tensor Train and Tensor Ring layers
# from Tensortrain import TensorTrainLayer #moved to fullintegration due to circular dependancy
# from Tensorring import TensorRingLayer #moved to fullintegration due to circular dependancy

# Define dimensions and rank (or import them)
input_dim = 512
output_dim = 1024
rank = 8

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# from Tensortrain import TensorTrainLayer #needs to be imported after OptimizedTensorNetwork
# from Tensorring import TensorRingLayer #needs to be imported after OptimizedTensorNetwork

# Define the TensorTrainLayer and TensorRingLayer classes here to avoid circular dependency
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

class OptimizedTensorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, rank, method="TT"):
        super(OptimizedTensorNetwork, self).__init__()
        self.method = method

        if method == "TT":
            self.layer = TensorTrainLayer(input_dim, output_dim, rank)
        elif method == "TR":
            self.layer = TensorRingLayer(input_dim, output_dim, rank)
        else:
            raise ValueError("Unsupported tensor decomposition method.")

    def forward(self, x):
        return self.layer(x)


# Move the model instantiation after the definitions of input_dim, output_dim, and rank
try:
    model = OptimizedTensorNetwork(input_dim, output_dim, rank, method="TT").to(device)
except Exception as e:
    print(f"Error creating model: {e}")
    model = None

if model:
    # Example Usage (moved inside the 'if model' block)
    x = torch.randn(64, input_dim, rank).to(device)
    with torch.no_grad():
        output = model(x)

    print("Inference Completed. Output Shape:", output.shape)