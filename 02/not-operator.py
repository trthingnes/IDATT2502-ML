import torch
import matplotlib.pyplot as plt

learning_rate = 1
epochs = 100


class SigmoidModel:
    def __init__(self):
        self.W = torch.rand((1, 1), requires_grad=True)
        self.b = torch.rand((1, 1), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


x_train = torch.tensor([[0.0], [1.0]])
y_train = torch.tensor([[1.0], [0.0]])

model = SigmoidModel()
optimizer = torch.optim.SGD([model.b, model.W], lr=learning_rate)

for epoch in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"W = {model.W}, b = {model.b}, loss = {model.loss(x_train, y_train)}")

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0, 1.0, 0.01).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = \sigma(xW + b)$')
plt.legend()
plt.show()


