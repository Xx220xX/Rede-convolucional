from gab_py_c import ManageTrain, Cnn, c,clib

luafile = 'D:/Henrique/treino_ia/treino_numero_0_9/config_09.lua'

cnn = Cnn(3, 3, 1)
cnn.convolucao(1,1,2)
print(cnn.sizeIn.x)
print(cnn.sizeIn.y)
print(cnn.sizeIn.z)
print(cnn.size)
print(cnn.L)
