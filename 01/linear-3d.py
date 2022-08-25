import torch
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.0001
epochs = 1_000_000


class LinearRegressionModel:
    def __init__(self):
        self.W = torch.rand((2, 1), requires_grad=True)
        self.b = torch.rand((1, 1), requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


data = np.loadtxt("day_length_weight.csv", delimiter=",", dtype="f")

train_x = torch.from_numpy(data[:, 1:]).reshape(-1, 2)  # Length, weight
train_y = torch.from_numpy(data[:, 0]).reshape(-1, 1)  # Day

model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.b, model.W], lr=learning_rate)

for epoch in range(epochs):
    model.loss(train_x, train_y).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"W = {model.W}, b = {model.b}, loss = {model.loss(train_x, train_y)}")

# Visualize result
ax = plt.figure().add_subplot(projection="3d")

ax.scatter(train_x[:, 0], train_x[:, 1], train_y, 'o', label='$(x^{(i)},y^{(i)}, z^{(i)})$')
ax.scatter(train_x[:, 0], train_x[:, 1], model.f(train_x).detach(), color="orange", label="$\\hat y = f(x) = xW+b$")

ax.set_xlabel("x: Length")
ax.set_ylabel("y: Weight")
ax.set_zlabel("z: Day")

plt.show()
