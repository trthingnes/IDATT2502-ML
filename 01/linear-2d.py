import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set hyper-parameters
lr = 1e-7
n_epochs = 3000

# Load data from file
data = np.loadtxt("length_weight.csv", delimiter=",", dtype="f")
x_tensor = torch.from_numpy(data[:, 0]).reshape(-1, 1)
y_tensor = torch.from_numpy(data[:, 1]).reshape(-1, 1)

# Build a sequential model with a single linear layer
model = nn.Linear(1, 1)
print(model.state_dict())

# Define loss and optimizer functions
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=lr)

losses = []


def train_step(x_values, y_values):
    model.train()

    # Find predicted value
    yhat = model(x_values)

    # Determine loss comparing actual y and yhat
    loss = loss_fn(y_values, yhat)
    loss.backward()

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


for epoch in range(n_epochs):
    loss = train_step(x_tensor, y_tensor)
    losses.append(loss)

# Print losses
print(np.array(losses))
print(f"Loss: { losses[-1] }")

# Visualize result
plt.plot(x_tensor, y_tensor, 'o')
plt.xlabel('x (length)')
plt.ylabel('y (weight)')

x = torch.tensor([[torch.min(x_tensor)], [torch.max(x_tensor)]])
y = model.to("cpu")(x).detach()
plt.plot(x, y, label='$\\hat y = f(x) = xW+b$')

plt.legend()
plt.show()
