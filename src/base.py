"""
Standard solution for Fizz Buzz
"""
from typing import Iterator


def fizz_buzz(n: int) -> list[str]:
    """
    Given an integer n, return a string array answer (1-indexed) where:
      answer[i] == "Fizzbuzz" if i is divisible by 3 and 5
      answer[i] == "Fizz" if i is divisible by 3
      answer[i] == "Buzz" if i is divisible by 5
      answer[i] == i (as a string) by none of the above conditions are true
    """
    answer = []
    buzz_map = {3: 'Fizz', 5: 'Buzz'}

    for i in range(1, n + 1):
        temp = ''
        for key, val in buzz_map.items():
            if i % key == 0:
                temp += val
        if not temp:
            temp = str(i)
        answer.append(temp)

    return answer


def fizz_buzz_improved(n: int) -> Iterator[str]:
    """
    A faster solution using generator
    """
    buzz_map = {3: 'Fizz', 5: 'Buzz'}

    for i in range(1, n + 1):
        temp = []
        for key, val in buzz_map.items():
            if i % key == 0:
                temp.append(val)
        yield ''.join(temp) if temp else str(i)


if __name__ == '__main__':
    N = int(1e7)
    fizz_buzz(N)
    fizz_buzz_improved(N)
