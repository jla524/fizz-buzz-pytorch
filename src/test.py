"""
Define test cases for fizz buzz
"""
from unittest import TestCase
import base
import model


class FizzBuzzTest(TestCase):
    """
    Test cases for list implmentations of fizz buzz
    """
    def test(self):
        """
        Test fizz buzz for cases up to 100
        """
        def validate(func, n):
            answer = func(n)
            for i in range(1, n + 1):
                if i % 15 == 0:
                    expected = 'FizzBuzz'
                elif i % 3 == 0:
                    expected = 'Fizz'
                elif i % 5 == 0:
                    expected = 'Buzz'
                else:
                    expected = str(i)
                self.assertEqual(answer[i-1], expected)
        validate(base.fizz_buzz, 100)
        validate(model.fizz_buzz, 100)
