import torch
import torch.nn as nn
import torch.optim as optim

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# XOR Dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

# Define a simple feed-forward network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 4),  # Input layer -> Hidden layer
            nn.ReLU(),        # Activation function
            nn.Linear(4, 1),  # Hidden layer -> Output layer
            nn.Sigmoid()      # Sigmoid to output probabilities
        )
    
    def forward(self, x):
        return self.layer(x)

# Initialize the model, loss function, and optimizer
model = XORNet().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 10000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Zero gradients
    output = model(X)      # Forward pass
    loss = criterion(output, y)  # Compute loss
    loss.backward()        # Backward pass
    optimizer.step()       # Update weights
    
    # Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model (Inference)
model.eval()
with torch.no_grad():
    predictions = model(X)
    predictions = (predictions > 0.5).float()  # Convert probabilities to binary predictions
    print("\nPredictions:")
    print(predictions.cpu().numpy())
