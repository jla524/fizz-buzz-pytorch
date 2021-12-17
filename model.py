"""
Solve fizz buzz using pytorch
"""
import numpy as np
import torch
from torch import nn, optim

epochs = 100
bs = 8
lr = 0.001


def encode_input(x):
    binary = []
    for _ in range(10):
        binary.append(x % 2)
        x //= 2
    return np.array(binary)


def encode_fizz_buzz(x):
    # Ground truth
    if x % 15 == 0:
        out = 3
    elif x % 5 == 0:
        out = 2
    elif x % 3 == 0:
        out = 1
    else:
        out = 0
    return out


def generate_data():
    # Train on 101 - 1023
    x_train = np.array([encode_input(i) for i in range(101, 1024)])
    y_train = np.array([encode_fizz_buzz(i) for i in range(101, 1024)])
    # Test on 1 - 100
    x_test = np.array([encode_input(i) for i in range(1, 101)])
    y_test = np.array([encode_fizz_buzz(i) for i in range(1, 101)])
    return x_train, y_train, x_test, y_test


class BuzzNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 100)
        self.l2 = nn.Linear(100, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x


def train(model, loss_fn, optimizer, x_train, y_train):
    accuracies = []
    losses = []
    for _ in range(epochs):
        samp = np.random.randint(0, len(x_train), size=(bs))
        x = torch.tensor(x_train[samp]).float()
        y = torch.tensor(y_train[samp])
        out = model(x)
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == y).float().mean()
        loss = loss_fn(out, y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss {loss} accuracy {accuracy}")
        accuracies.append(accuracy.item())
        losses.append(loss.item())
    return accuracies, losses


def fizz_buzz(n):
    answer = []
    for i in range(1, n + 1):
        options = [str(i), 'Fizz', 'Buzz', 'FizzBuzz']
        answer.append(options[1])
    return answer


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = generate_data()
    net = BuzzNet()
    loss_function = nn.NLLLoss(reduction='none')
    optim = optim.SGD(net.parameters(), lr=lr)
    # TODO: Accuracy is very low
    train(net, loss_function, optim, x_train, y_train)
