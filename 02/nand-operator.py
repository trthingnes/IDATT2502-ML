import torch
import matplotlib.pyplot as plt

learning_rate = 1
epochs = 500


class SigmoidModel:
    def __init__(self):
        self.W = torch.rand((2, 1), requires_grad=True)
        self.b = torch.rand((1, 1), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

model = SigmoidModel()
optimizer = torch.optim.SGD([model.b, model.W], lr=learning_rate)

for epoch in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"W = {model.W}, b = {model.b}, loss = {model.loss(x_train, y_train)}")

# Visualize result
ax = plt.figure().add_subplot(projection="3d")

size = 10
x, y = torch.meshgrid(torch.arange(0, 1, 1/size), torch.arange(0, 1, 1/size))
z = torch.zeros([size, size])

for row in range(size):
    for col in range(size):
        z[row, col] = model.f(torch.tensor([x[row, col], y[row, col]])).detach()

ax.plot_surface(x, y, z, color="blue")

plt.show()



