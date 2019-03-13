"""
This class handles all of the symbolic math, including computing terms for series expansions.
Mainly, it spits out terms of expansion evaluation.
Expansion evaluation involves taking the Tayler series expansion of the function about a point.
We then evaluate the series at a certain point and spit out the results term by term.
"""
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import z
from sympy import diff
import numpy as np

class Function:
    def __init__(self, string):
        try:
            self.expr = parse_expr(string)
        except:
            print('Please input a valid function.')
            1/0
        self.term_gen = None
        self.partial_sum = 0.0

    def make_term_gen(self, expr, difference, a):
        """
        Creates a generator for the consecutive terms of the expansion evaluation of the function.
        """
        diff_power = 1
        f_prime = expr
        n = 0
        factorial = 1
        while True:
            output = (f_prime / factorial)*diff_power
            f_prime = diff(f_prime, z)
            diff_power = diff_power * difference
            n = n + 1
            factorial = factorial * n
            yield complex(output.subs({z:a}).evalf())


    def expand_eval(self, a, z0):
        """
        Compute series expansion of the function about the point "a", and evaluated at "z0".
        This actually creates a generator for the terms.
        """
        difference = z0 - a
        self.term_gen = self.make_term_gen(self.expr, difference, a)

    def next_term(self):
        """
        Returns the next term in the expansion evaluation.
        If evaluation and expansion points have not been chosen, we return None.
        """
        if (self.term_gen == None):
            print("You need to set expansion and evaluation points.")
            return None
        else:
            term = np.complex(next(self.term_gen))
            self.partial_sum += term
            return term
