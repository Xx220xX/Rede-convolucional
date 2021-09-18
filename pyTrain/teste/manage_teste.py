import random

from gab_py_c import *

luafile = 'D:/Henrique/treino_ia/treino_numero_0_9/config_09.lua'
# SetSeed(time.time())
SetSeed(50)
cnn = Cnn(3, 3, 1)
cnn.convolucao(1,2,6)
cnn.FullConnect(2)
cnn.call(list(range(-3,6)))
cnn.save('rede0.cnn')
print(cnn.camadas[1].saida.getValues(cnn.queue))
del cnn
cnn2 = Cnn.load('rede0.cnn')
cnn2.call(list(range(-3,6)))
print(cnn2.camadas[1].saida.getValues(cnn2.queue))

