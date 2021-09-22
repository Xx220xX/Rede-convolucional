import random

from gab_py_c import *

luafile = 'D:/Henrique/treino_ia/treino_numero_0_9/config_09.lua'

# SetSeed(time.time())
usePythread()

@EVENT
def update(t: Manage_p):
	mn: ManageTrain
	mn = c.cast(t, c.POINTER(ManageTrain))[0]
	print(mn.et.ll_imagem_atual)


data = open(luafile, 'r').read()
manage = ManageTrain(data, luaisFile=False)
print(manage.n_images)
print(manage.n_images2train)
print(manage.n_images2fitness)
manage.setEvent(manage.UpdateLoad, update)
manage.loadImageStart()
manage.startLoop(True)

while manage.et.ll_imagem_atual < manage.n_images - 1:
	pass
