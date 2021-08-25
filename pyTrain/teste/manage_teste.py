from gab_py_c import ManageTrain, Cnn, c

luafile = 'D:/Henrique/treino_ia/treino_numero_0_9/config_09.lua'

print('main', c.sizeof(Cnn))

cnn = Cnn(3, 3, 3)
p = cnn.address()
print(p[0])
cnn.__del__()