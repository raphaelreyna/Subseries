from __future__ import absolute_import
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
import sys
from Function import *
import numpy as np
from SharedDevice import *

class Renderer:
    def __init__(self, k):
        self.k = k
        self.iterations = 0
        self.point_count = 2**self.k
        self.anchor_path = os.path.abspath(os.path.dirname(__file__))
        self.f = Function('exp(z)')
        self.f.expand_eval(0, 1+1.5j)
        self.setupGlut()
        self.sd = SharedDevice(self.vbo, self.k)
        self.compute()

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_POINTS, 0, self.point_count)
        glFlush()

    def reshape(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)

    def setupGlut(self):
        glutInit([])
        glutInitWindowSize(800, 800)
        glutInitWindowPosition(0, 0)
        glutCreateWindow('Subserial')
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glClearColor(1, 1, 1, 1)
        glColor(0, 0, 1)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.point_count * 2 * 4, None, GL_STATIC_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, None)

    def compute(self):
        for i in range(0, 30):
            self.sd.process_next_term(self.f.next_term())

    def run(self):
        glutMainLoop()

if __name__ == '__main__':
    r = Renderer(15)
    r.run()
