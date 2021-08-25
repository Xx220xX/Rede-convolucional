import ctypes as c
import os
from os.path import isfile
import sys

__dir__ = os.path.abspath(__file__)
__dir__ = os.path.realpath(__dir__)
__dir__ = os.path.dirname(__dir__)
__dir__ = os.path.join(__dir__, '../bin/')
__dir__ = os.path.realpath(__dir__)
if __dir__ not in sys.path:
	sys.path.append(__dir__)
__dll__ = 'libCNNGPU.dll'
for path in sys.path:
	if isfile(path+'/'+__dll__):
		__dll__ = path+'/'+__dll__
		break

clib = c.CDLL(__dll__)

