import random

from gab_py_c import *

luafile = 'D:/Henrique/treino_ia/treino_numero_0_9/config_09.lua'


# SetSeed(time.time())

@EVENT
def upload(t: Manage_p):
	mn: ManageTrain
	mn = t[0]
	print(mn.et.ll_imagem_atual)

data = open(luafile,'r').read()
manage = ManageTrain(data,luaisFile=False)
print(manage.n_images)
print(manage.n_images2train)
print(manage.n_images2fitness)
# manage.setEvent(manage.UpdateLoad, upload)
# manage.loadImageStart()
# manage.startLoop()
