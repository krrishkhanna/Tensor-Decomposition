import torch
import torch.optim as optim
import torch.nn as nn

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dimensions and rank
input_dim = 512  # Example value
output_dim = 1024  # Example value
rank = 8  # Example value


def train_tensorized_model(model, data_loader, epochs=5, lr=1e-3):
    """
    Trains a given PyTorch model using the provided data loader.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): The data loader for training data.
        epochs (int): The number of epochs to train for.
        lr (float): The learning rate for the optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()  # Set model to training mode
    for epoch in range(epochs):
        for batch in data_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # Reset gradients
            output = model(x)  # Forward pass
            loss = loss_fn(output, y)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Dummy dataset (replace with your actual data loading)
data_loader = [(torch.randn(64, input_dim, rank), torch.randn(64, output_dim, rank)) for _ in range(100)]

# from optimisinginferencetime import OptimizedTensorNetwork #Importing here causes circular dependancy
# model = OptimizedTensorNetwork(input_dim, output_dim, rank, method="TT").to(device) #requires Tensorring.py and Tensortrain.py which in turn require modeltensorint.py in this situation will throw an error.


# train_tensorized_model(model, data_loader) #removed this line to test as this requires optimisinginferencetime.py and throws error