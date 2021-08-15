try:
	from gab_wrapper_structs import *
	from gab_wrapper_load_dll import *
except Exception:
	from gab_py_c.gab_wrapper_structs import *
	from gab_py_c.gab_wrapper_load_dll import *
import numpy as np


class ManageTrain(ManageTrain):
	def chose2WorkDir(self):
		clib.manage2WorkDir(c.addressof(self))

	def __del__(self):
		clib.releaseManageTrain(c.addressof(self))

	def setEvent(self, self_event, event):
		clib.manageTrainSetEvent(c.addressof(self_event), event)

	def setRun(self, can_run):
		can_run = int(can_run)
		clib.manageTrainSetRun(c.addressof(self), can_run)

	def __init__(self, luafile, taxa_aprendizado=0.1, momento=0, decaimento_peso=0):
		super().__init__()
		clib.createManageTrainPy(c.addressof(self), luafile.encode('utf-8'), taxa_aprendizado, momento, decaimento_peso)

	def loadImageStart(self):
		clib.ManageTrainloadImages(c.addressof(self))

	def trainStart(self):
		clib.ManageTraintrain(c.addressof(self))

	def fitnesStart(self):
		clib.ManageTrainfitnes(c.addressof(self))

	def startLoop(self, anotherThread=True):
		anotherThread = int(anotherThread)
		clib.manageTrainLoop(c.addressof(self), anotherThread)

	def save(self, file_name):
		file_name = file_name.encode('utf-8')
		return clib.CnnSaveInFile(c.addressof(self), file_name)


class Camada(Camada):
	def __repr__(self):
		t = self.toString(c.addressof(self))
		t = t.value.decode('utf-8')
		return t


class Cnn(Cnn):
	def __repr__(self):
		t = []
		for l in range(self.size):
			t.append(str(self.camadas[l]))
		return '\n'.join(t)


class Tensor(Tensor):
	def getvalues(self, ):
		pass

	def getvalues_np(self):
		pass