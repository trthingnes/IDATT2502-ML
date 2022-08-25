import torch
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.0000005
epochs = 5000
print_every = 1000


class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


data = np.loadtxt("day_head_circumference.csv", delimiter=",", dtype="f")

train_x = torch.from_numpy(data[:, 0]).float().reshape(-1, 1)  # Day
train_y = torch.from_numpy(data[:, 1]).float().reshape(-1, 1)  # Head circumference

model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.b, model.W], lr=learning_rate)


for epoch in range(epochs // print_every):
    for _ in range(print_every):
        model.loss(train_x, train_y).backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"{epoch * print_every}: Loss is {model.loss(train_x, train_y)}")

print(f"W = {model.W}, b = {model.b}, loss = {model.loss(train_x, train_y)}")

# Visualize result
plt.title("Head circumference determined by age")
plt.xlabel("x: Age [days]")
plt.ylabel("y: Head circumference [cm]")

plt.scatter(train_x, train_y)
x_coords = torch.arange(torch.min(train_x), torch.max(train_x)).reshape(-1, 1)
y_coords = model.f(x_coords).detach()
plt.plot(x_coords, y_coords, color="red", label="$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$")

plt.legend()
plt.show()
