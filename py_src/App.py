"""
This class handles user interaction and event loop.
"""
import argparse, sys, os
from Function import Function
from Renderer import Renderer

class App:
    def __init__(self, args=None):
        self.func = None
        self.image = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('function', help = 'Function you wish to work with. Must be holomorphic on D(0,1) as a function of z.')
        self.parser.add_argument('degree', help='Degree to which we shall expand the function.')
        self.parser.add_argument('a', help='Base point for expansion. Do not exceed the open unit circle, results may be non-sense.')
        self.parser.add_argument('z0', help='Point at which to perform spectral evaluation.')

    def main(self):
        args = self.parser.parse_args(sys.argv[1:])
        self.func = Function(args.function)
        self.renderer = Renderer()
        self.func.expand_eval(complex(args.z0), complex(args.a))
        points = []
        for i in range(0, int(args.degree)):
            points.append(self.func.next_term())
        self.renderer.addPoints(points)
        filename = 'spectralEval'+args.function+'.png'
        self.renderer.image.save('output.png','PNG')
        print('Saved .png to this directory.')
        self.renderer.dump()
        print('Dumped points as .json file in this directory.')
