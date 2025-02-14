import time
import torch

# Assuming device, model, input_dim, and rank are defined elsewhere
# If not, define them here:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 512  # Example
output_dim = 1024  # Example
rank = 8  # Example


def benchmark_model(model, input_tensor, runs=100):
    """
    Benchmarks the given PyTorch model's inference time.

    Args:
        model (torch.nn.Module): The model to benchmark.
        input_tensor (torch.Tensor): The input tensor for the model.
        runs (int): The number of inference runs to perform.
    """
    model.eval()  # Set model to evaluation mode
    start_time = time.time()
    memory_before = torch.cuda.memory_allocated()
    max_memory_before = torch.cuda.max_memory_allocated()

    with torch.no_grad():  # Disable gradient calculation
        for _ in range(runs):
            _ = model(input_tensor)

    end_time = time.time()
    memory_after = torch.cuda.memory_allocated()
    max_memory_after = torch.cuda.max_memory_allocated()

    print(f"Inference Time: {(end_time - start_time) / runs:.6f} sec per run")
    print(f"Memory Allocated: {memory_after - memory_before} bytes")
    print(f"Max Memory Allocated: {max_memory_after - max_memory_before} bytes")


# Benchmarking before and after optimization
# Need to define model before this will work
# input_tensor = torch.randn(64, input_dim, rank).to(device)
# benchmark_model(model, input_tensor)