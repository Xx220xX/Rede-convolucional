from gabriela_gpu import DNN
from gabriela_gpu.dnn_gpu import setSeed

setSeed(1)
a = DNN((1, 3, 3, 1))
a([1])
print(a.out)
a.aprender([2])
a([1])
print(a.out)
setSeed(1)
a.randomize()
a([1])
print(a.out)
