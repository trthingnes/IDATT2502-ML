import torch
import torch.nn as nn
import torchvision

# 2 conv-64-128 = 0.87
# 2 conv-32-65 1/6 batch = 0.88
# 2 conv-32-64 = 0.86
# 2 conv-32-64, dropout-0.1 = 0.85
# 2 conv-32-64, dropout-0.5 = 0.82
# 2 conv-32-64, dropout-0.5, relu = 0.81
# 2 conv-32-64, relu before maxpool = 0.89


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14 * 14

            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 7 * 7

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 10)
        )

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)
y_test = torch.zeros((mnist_test.targets.shape[0], 10)).to(device)
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 1000
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

model = ConvolutionalNeuralNetworkModel()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test))
