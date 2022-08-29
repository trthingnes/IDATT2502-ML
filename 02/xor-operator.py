import torch
import matplotlib.pyplot as plt

learning_rate = 1
epochs = 100


class SigmoidModel:
    def __init__(self):
        self.W1 = torch.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True)
        self.b1 = torch.tensor([[-5.0, 15.0]], requires_grad=True)
        self.W2 = torch.tensor([[10.0], [10.0]], requires_grad=True)
        self.b2 = torch.tensor([[-15.0]], requires_grad=True)

    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self, x):
        return torch.sigmoid(x @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = SigmoidModel()
optimizer = torch.optim.SGD([model.b1, model.W1, model.b2, model.W2], lr=learning_rate)

for epoch in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"W1 = {model.W1}, b1 = {model.b1}, W2 = {model.W2}, b2 = {model.b2}, loss = {model.loss(x_train, y_train)}")

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



