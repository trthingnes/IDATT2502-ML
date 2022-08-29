import os
import torch
import torchvision
import matplotlib.pyplot as plt

learning_rate = 0.00003
epochs = 1000
print_every = 100


def get_mnist_data(train):
    mnist = torchvision.datasets.MNIST("./data", train=train, download=True)
    x = mnist.data.reshape(-1, 784).float()
    y = torch.zeros((mnist.targets.shape[0], 10))
    y[torch.arange(mnist.targets.shape[0]), mnist.targets] = 1

    return x, y


class MnistModel:
    def __init__(self):
        self.W = torch.ones((784, 10), requires_grad=True)
        self.b = torch.ones((1, 10), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.nn.functional.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


x_train, y_train = get_mnist_data(train=True)
x_test, y_test = get_mnist_data(train=False)

model = MnistModel()
optimizer = torch.optim.SGD([model.b, model.W], lr=learning_rate)

for epoch in range(epochs // print_every):
    for _ in range(print_every):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"{(epoch + 1) * print_every}: loss = {model.loss(x_train, y_train)}   acc = {model.accuracy(x_test, y_test)}")

print(f"loss = {model.loss(x_train, y_train)}   acc = {model.accuracy(x_test, y_test)}")

# Visualize result
os.mkdir("result")
for i in range(10):
    plt.imsave(f"result/{i}.png", model.W[:, i].reshape(28, 28).detach())
