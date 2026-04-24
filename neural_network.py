# Import libraries
# numpy is used for handling data
# torch is used to build and train the neural network
import numpy as np  # Import NumPy for array operations
import torch  # Import PyTorch for neural networks
import torch.nn as nn  # Import neural network modules from PyTorch


# -------------------------------
# INPUT DATA
# -------------------------------
# These are the inputs to the neural network
# Each row is one example with two inputs
X = np.array([  # Create input data as NumPy array
    [0, 0],  # First input: both bits 0
    [0, 1],  # Second input: first bit 0, second bit 1
    [1, 0],  # Third input: first bit 1, second bit 0
    [1, 1]   # Fourth input: both bits 1
])

# Correct outputs for the inputs
Y = np.array([  # Create target outputs as NumPy array
    [0],  # Output for [0,0]: 0 (XOR result)
    [1],  # Output for [0,1]: 1
    [1],  # Output for [1,0]: 1
    [0]   # Output for [1,1]: 0
])

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)  # Convert inputs to PyTorch float tensor
Y = torch.tensor(Y, dtype=torch.float32)  # Convert targets to PyTorch float tensor


# -------------------------------
# BUILD THE NEURAL NETWORK
# -------------------------------

# Sequential means layers are connected one after another
model = nn.Sequential(  # Create a sequential neural network model
    nn.Linear(2, 8),  # First linear layer: 2 inputs to 8 neurons
    nn.ReLU(),        # ReLU activation function for first layer
    nn.Linear(8, 4),  # Second linear layer: 8 neurons to 4 neurons
    nn.ReLU(),        # ReLU activation function for second layer
    nn.Linear(4, 1),  # Output linear layer: 4 neurons to 1 neuron
    nn.Sigmoid()      # Sigmoid activation for binary output (0-1)
)


# -------------------------------
# COMPILE MODEL
# -------------------------------
# optimizer adjusts weights and bias during training
# loss measures prediction error
loss_fn = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
optimizer = torch.optim.Adam(model.parameters())  # Adam optimizer for updating model parameters


# -------------------------------
# TRAIN THE MODEL
# -------------------------------
# epochs = number of times the model learns from the dataset
# During training:
# 1. inputs go through neurons
# 2. neurons apply weights and bias
# 3. activation functions produce outputs
# 4. error is calculated
# 5. weights and bias are updated
for epoch in range(1000):  # Loop over 1000 training epochs
    # Forward pass
    outputs = model(X)  # Pass inputs through the model to get predictions
    loss = loss_fn(outputs, Y)  # Calculate loss between predictions and targets
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients via backpropagation
    optimizer.step()  # Update model parameters using gradients
    
    if (epoch + 1) % 200 == 0:  # Print loss every 200 epochs
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')  # Display current epoch and loss


# -------------------------------
# MAKE PREDICTIONS
# -------------------------------
# The trained model predicts outputs
with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(X).numpy()  # Get predictions and convert to NumPy array

print("Predictions:")  # Print header for predictions
print(predictions)  # Print the predicted values


# -------------------------------
# VIEW LEARNED WEIGHTS AND BIAS
# -------------------------------
# Each neuron has weights and bias
weights = model[0].weight.data.numpy()  # Get weights of first layer as NumPy array
bias = model[0].bias.data.numpy()  # Get bias of first layer as NumPy array

print("\nFirst layer weights:")  # Print header for weights
print(weights)  # Print the weight matrix

print("\nFirst layer bias:")  # Print header for bias
print(bias)  # Print the bias vector