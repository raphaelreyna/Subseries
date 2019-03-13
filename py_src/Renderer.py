"""
This class handles displaying the generated data.
"""
from PIL import Image, ImageDraw
from itertools import starmap
from functools import reduce

class Renderer:
    def __init__(self, points = None, size = (500,500), bg = (255,255,255,255)):
        self.points = points
        self.image = Image.new('RGBA', size, bg)
        self.ctx = ImageDraw.Draw(self.image)

    def addPoints(self, newPoints):
        average = reduce(lambda x,y: x+y, newPoints)/len(newPoints)
        print(average)
        centeredPoints = map(lambda x: x - average, newPoints)
        maxMod = max([abs(point) for point in centeredPoints])
        print(maxMod)
        scaledPoints = map(lambda x: (240/maxMod)*x, newPoints)
        self.points = [(z.real+250, z.imag+350) for z in scaledPoints]
        self.ctx.point(self.points, fill=(255,0,0,200))

    def dump(self):
        file = open('dump.json', 'w')
        json.dump(self.points, file)
        file.close()
