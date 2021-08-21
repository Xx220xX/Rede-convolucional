from gab_wrapper_load_dll import *
import numpy as np


class ManageTrain(c.Structure):
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


class Camada(c.Structure):
	def __repr__(self):
		t = self.toString(c.addressof(self))
		t = t.value.decode('utf-8')
		return t


class Cnn(c.Structure):
	def __repr__(self):

		t = [f'CNN {c.addressof(self)} size {self.size}']
		try:
			for l in range(self.size):
				t.append(str(self.camadas[l]))
		except: pass
		return '\n'.join(t)
	def __init__(self):
		super(Cnn, self).__init__()
		print('here')


class Tensor(c.Structure):
	def getvalues(self, ):
		pass

	def getvalues_np(self):
		pass

class String(c.Structure):
	pass


class Dbchar_p(c.Structure):
	pass


class List_args(c.Structure):
	pass


class Ponto(c.Structure):
	pass



class Kernel(c.Structure):
	pass


class CNN_ERROR(c.Structure):
	pass


class Params(c.Structure):
	pass




class CamadaBatchNorm(c.Structure):
	pass


class CamadaConv(c.Structure):
	pass


class CamadaConvNc(c.Structure):
	pass


class CamadaDropOut(c.Structure):
	pass


class CamadaFullConnect(c.Structure):
	pass


class CamadaPadding(c.Structure):
	pass


class CamadaPool(c.Structure):
	pass


class CamadaPoolAv(c.Structure):
	pass


class CamadaRelu(c.Structure):
	pass


class CamadaSoftMax(c.Structure):
	pass




class Estatistica(c.Structure):
	pass

