
import ctypes as c
import os
__dir = os.path.abspath(__file__)
__dir = os.path.realpath(__dir)
__dir = os.path.dirname(__dir)
__dll = os.path.join(__dir, '../bin/clib')

clib = c.CDLL(__dll)

