import random

from gab_py_c import *

luafile = 'D:/Henrique/treino_ia/treino_numero_0_9/config_09.lua'

# SetSeed(time.time())
usePythread()


@EVENT
def F(t: Manage_p):
	print('Leu')
	mn: ManageTrain
	mn = c.cast(t, c.POINTER(ManageTrain))[0]
	print('end', mn.et.ll_imagem_atual)


@EVENT
def update(t: Manage_p):
	print('Here')
	mn: ManageTrain
	mn = c.cast(t, c.POINTER(ManageTrain))[0]


data = open(luafile, 'r').read()
manage = ManageTrain(data, luaisFile=False)

# manage.setEvent(manage.UpdateLoad, update)
manage.setEvent(manage.OnFinishLoop, F)
# manage.loadImageStart(True)
#
# manage.startLoop(True)
#
print(int(manage.cnn.camadas[0].type[0]))
