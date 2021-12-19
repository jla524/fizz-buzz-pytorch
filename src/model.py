"""
Solve fizz buzz using pytorch
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import trange

weights_path = Path('../models/weights')


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
        self.l1 = nn.Linear(10, 200)
        self.l2 = nn.Linear(200, 200)
        self.l3 = nn.Linear(200, 4)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.dropout(self.l3(x))
        return x


def train(model, x_train, y_train):
    iterations = 7000
    size = 128
    rate = 0.003
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=rate)
    model.train()
    for _ in (t := trange(iterations)):
        samp = np.random.randint(0, len(x_train), size=(size))
        x = torch.tensor(x_train[samp]).float()
        y = torch.tensor(y_train[samp])
        out = model(x)
        loss = loss_func(out, y)
        cat = torch.argmax(model(x), dim=1)
        accuracy = (cat == y).float().mean()
        t.set_description(f"loss = {loss} accuracy = {accuracy}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, x_test, y_test):
    model.eval()
    x = torch.tensor(x_test).float()
    cat = torch.argmax(model(x), dim=1).numpy()
    accuracy = (cat == y_test).mean()
    print(f"accuracy = {accuracy}")
    if accuracy == 1.0:
        torch.save(model.state_dict(), weights_path)
    return accuracy


def get_model():
    net = BuzzNet()
    if weights_path.is_file():
        net.load_state_dict(torch.load(weights_path))
        net.eval()
    else:
        x_train = np.array([encode_input(i) for i in range(101, 1024)])
        y_train = np.array([encode_fizz_buzz(i) for i in range(101, 1024)])
        x_test = np.array([encode_input(i) for i in range(1, 101)])
        y_test = np.array([encode_fizz_buzz(i) for i in range(1, 101)])
        train(net, x_train, y_train)
        test(net, x_test, y_test)
    return net


def fizz_buzz(n):
    answer = []
    model = get_model()
    for i in range(1, n + 1):
        options = [str(i), 'Fizz', 'Buzz', 'FizzBuzz']
        x = torch.tensor(encode_input(i)).reshape(1, 10).float()
        cat = torch.argmax(model(x), dim=1)
        answer.append(options[cat])
    return answer


if __name__ == '__main__':
    print(fizz_buzz(100))
