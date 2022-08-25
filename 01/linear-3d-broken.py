import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set hyper-parameters
lr = 1e-4
n_epochs = 10000

# Load data from file
data = np.loadtxt("day_length_weight.csv", delimiter=",", dtype="f")
x_tensor = torch.from_numpy(data[:, 1:]).reshape(-1, 2)  # Length, weight
y_tensor = torch.from_numpy(data[:, 0]).reshape(-1, 1)  # Day

# Build a sequential model with a single linear layer
model = nn.Linear(2, 1)
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

print(f"x: 10, y: 80, z: {model(torch.tensor([[10, 80]]).float())}")

# Visualize result
ax = plt.figure().add_subplot(projection="3d")

ax.scatter(x_tensor[:, 0], x_tensor[:, 1], y_tensor, 'o')

ax.plot([10], [80], model(torch.tensor([[10, 80]]).float()).detach()[0], color="orange")

ax.set_xlabel("x: Length")
ax.set_ylabel("y: Weight")
ax.set_zlabel("z: Day")

plt.show()
