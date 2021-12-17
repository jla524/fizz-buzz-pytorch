"""
Solve fizz buzz using pytorch
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

epochs = 10000
bs = 128
lr = 0.01


def encode_input(x):
    return np.array([x >> d & 1 for d in range(10)])


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


class BuzzNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 50)
        self.l2 = nn.Linear(50, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


def train(model, x_train, y_train):
    loss_func = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        samp = np.random.randint(0, len(x_train), size=(bs))
        x = torch.tensor(x_train[samp]).float()
        y = torch.tensor(y_train[samp])
        out = model(x)
        loss = loss_func(out, y)
        print(f"loss {loss}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, x_test, y_test):
    x = torch.tensor(x_train).float()
    cat = torch.argmax(model(x), dim=1).numpy()
    accuracy = (cat == y_test).mean()
    return accuracy


def fizz_buzz(model, n):
    answer = []
    for i in range(1, n + 1):
        options = [str(i), 'Fizz', 'Buzz', 'FizzBuzz']
        x = torch.tensor(encode_input(i)).reshape(1, 10).float()
        cat = torch.argmax(model(x), dim=1)
        answer.append(options[cat])
    return answer


if __name__ == '__main__':
    # Train on 101-1023 and test on 1-100
    x_train = np.array([encode_input(i) for i in range(101, 1024)])
    y_train = np.array([encode_fizz_buzz(i) for i in range(101, 1024)])
    x_test = np.array([encode_input(i) for i in range(1, 101)])
    y_test = np.array([encode_fizz_buzz(i) for i in range(1, 101)])

    # Train the neural network
    net = BuzzNet()
    train(net, x_train, y_train)
    print(test(net, x_train, y_train))
    print(fizz_buzz(net, 100))
