import torch
import torch.nn as nn
from transformers import AutoModel  # type: ignore

# Import TensorTrainLayer and TensorRingLayer *before* defining OptimizedTensorNetwork
try:
    from Tensorring import TensorRingLayer
    from Tensortrain import TensorTrainLayer
except ImportError as e:
    print(f"Error importing Tensorring or Tensortrain: {e}")
    TensorRingLayer = None
    TensorTrainLayer = None


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dimensions and rank
rank = 8  # Example rank
input_dim = 512  # example input_dim
output_dim = 1024  # example output_dim


# Define the OptimizedTensorNetwork class
class OptimizedTensorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, rank, method="TT"):
        super().__init__()  # Call super().__init__()
        self.method = method

        if TensorTrainLayer is not None and TensorRingLayer is not None:
            if method == "TT":
                self.layer = TensorTrainLayer(input_dim, output_dim, rank)
            elif method == "TR":
                self.layer = TensorRingLayer(input_dim, output_dim, rank)
            else:
                raise ValueError("Unsupported tensor decomposition method.")
        else:
            raise ValueError("TensorTrainLayer or TensorRingLayer could not be imported.")

    def forward(self, x):
        if not hasattr(self, 'layer'):
            raise RuntimeError("The layer attribute has not been initialized. This likely indicates a configuration error during model setup.")
        return self.layer(x)


# Load a pretrained LLM (LLaMA-2, GPT, etc.)
try:
    model = AutoModel.from_pretrained("meta-llama/Llama-2-7b").to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle the case where loading fails!

if model:  # only if the model is loaded then it can be modified
    # Replace dense layers with tensorized layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                setattr(
                    model,
                    name,
                    OptimizedTensorNetwork(module.in_features, module.out_features, rank, method="TT"),
                )
            except ValueError as e:
                print(f"Skipping layer {name} due to error: {e}")