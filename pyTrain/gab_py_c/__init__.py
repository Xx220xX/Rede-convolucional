import os
import sys

__dir = os.path.abspath(__file__)
__dir = os.path.realpath(__dir)
__dir = os.path.dirname(__dir)
sys.path.append(__dir)
sys.path.append('D:/Henrique/Rede-convolucional/CNN_GPU/bin')

from gab_cnn import *
