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


def encode_input(num: int) -> list[int]:
    """
    Converts the given number into a binary list (binary encoding)
    :param num: the number to convert
    :return: the encoded binary list
    """
    return np.array([num >> d & 1 for d in range(10)])


def encode_fizz_buzz(num: int) -> int:
    """
    Converts the given number into the expected case (ground truth)
    :param num: the number to convert
    :return: the encoded category
    """
    if num % 15 == 0:
        out = 3
    elif num % 5 == 0:
        out = 2
    elif num % 3 == 0:
        out = 1
    else:
        out = 0
    return out


class BuzzNet(nn.Module):
    """
    A class to represent the multi-layer-perceptron (MLP)
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 4)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, num: torch.Tensor) -> torch.Tensor:
        """
        Represents our forward pass
        :param num: the input data
        :return: the output data
        """
        num = F.relu(self.layer1(num))
        num = F.relu(self.layer2(num))
        num = self.dropout(self.layer3(num))
        return num


def train(model: type[BuzzNet], x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Trains the given model using training data
    :param model: the model to train
    :param x_train: the features to train on
    :param y_train: the expected outputs
    :return: None
    """
    iterations = 10000
    size = 128
    rate = 0.003
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=rate)
    model.train()
    for _ in (iteration := trange(iterations)):
        samp = np.random.randint(0, len(x_train), size=(size))
        x = torch.tensor(x_train[samp]).float()
        y = torch.tensor(y_train[samp])
        out = model(x)
        loss = loss_func(out, y)
        cat = torch.argmax(model(x), dim=1)
        accuracy = (cat == y).float().mean()
        iteration.set_description(f"loss = {loss} accuracy = {accuracy}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model: type[BuzzNet], x_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Tests the given model using test data
    :param model: the model to test
    :param x_test: the features to test
    :param y_test: the expected outputs
    :return: the accuracy of the model
    """
    model.eval()
    x = torch.tensor(x_test).float()
    cat = torch.argmax(model(x), dim=1).numpy()
    accuracy = (cat == y_test).mean()
    print(f"accuracy = {accuracy}")
    if accuracy == 1.0:
        torch.save(model.state_dict(), weights_path)
    return accuracy


def get_model() -> type[BuzzNet]:
    """
    Loads a trained multi-layer-perceptron (MLP) to solve Fizz Buzz
    :return: the trained MLP
    """
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


def fizz_buzz(num: int) -> list[str]:
    """
    Solves the Fizz Buzz problem
    :param num: the number of items in the answer
    :return: the answer
    """
    answer = []
    model = get_model()
    for i in range(1, num + 1):
        options = [str(i), 'Fizz', 'Buzz', 'FizzBuzz']
        x = torch.tensor(encode_input(i)).reshape(1, 10).float()
        cat = torch.argmax(model(x), dim=1)
        answer.append(options[cat])
    return answer


if __name__ == '__main__':
    print(fizz_buzz(100))
