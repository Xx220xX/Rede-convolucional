import ctypes as c
import os

__dir = os.path.abspath(__file__)
__dir = os.path.realpath(__dir)
__dir = os.path.dirname(__dir)
__dll = os.path.join(__dir, '..\\bin\\libCNNGPU.dll')
__dll = os.path.realpath(__dll)

# D:\Henrique\Rede-convolucional\pyTrain\bin\libCNNGPU.dll
clib = c.WinDLL('C:\\libCNNGPU.dll')
# clib = c.WinDLL(__dll)
print('here')
input()
