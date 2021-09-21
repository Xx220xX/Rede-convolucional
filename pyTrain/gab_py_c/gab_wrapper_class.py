import time

from gab_wrapper_load_dll import *
from gab_wrapper_functionspy import *
import numpy as np
import inspect

clib: LIBCNN


class CStruct(c.Structure):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

	def __repr__(self):
		return str(type(self))

	def address(self):
		P = c.addressof(self)
		return P


class Pointer(CStruct):
	# ${Pointer}
	pass


class RandomParam(CStruct):
	# ${RandomParam}
	pass


class Tensor(CStruct):
	# ${Tensor}
	def getValues(self, queue):
		tp = self.getType()
		t = (tp * len(self))(0)
		clib.TensorGetValuesMem(queue, self.address(), t, c.sizeof(t))
		return list(t)

	def getType(self):
		if (self.flag & 0b01100000) == 0b00100000:
			return c.c_char
		if (self.flag & 0b01100000) == 0b01000000:
			return c.c_int
		return c.c_double

	def getValues_np(self, queue):
		if self.w == 1:
			if self.z == 1:
				shape = (self.x, self.y)
			else:
				shape = (self.z, self.x, self.y)
		else:
			shape = (self.x, self.y, self.z, self.w)
		return np.array(self.getValues(queue)).reshape(shape)

	def __len__(self):
		return self.x * self.y * self.z * self.w

	def shape(self):
		return self.x, self.y, self.z, self.w

	def norm(self, queue):
		norm = c.c_double(0)
		erro = clib.TensorGetNorm(queue, self.address(), c.addressof(norm))
		if erro: raise Exception(f"Error code:{erro}")
		return norm.value


class String(CStruct):
	# ${String}
	pass


class Dbchar_p(CStruct):
	# ${Dbchar_p}
	pass


class Dictionary(CStruct):
	# ${Dictionary}
	pass


class Ponto(CStruct):
	# ${Ponto}
	pass


class Kernel(CStruct):
	# ${Kernel}
	pass


class CNN_ERROR(CStruct):
	# ${CNN_ERROR}
	pass


class Params(CStruct):
	# ${Params}
	pass


class Camada(CStruct):
	# ${Camada}
	def __repr__(self):
		t = self.toString(c.addressof(self))
		t = t.decode('utf-8')
		return t


class CamadaBatchNorm(CStruct):
	# ${CamadaBatchNorm}
	pass


class CamadaConv(CStruct):
	# ${CamadaConv}
	pass


class CamadaConvNc(CStruct):
	# ${CamadaConvNc}
	pass


class CamadaDropOut(CStruct):
	# ${CamadaDropOut}
	pass


class CamadaFullConnect(CStruct):
	# ${CamadaFullConnect}
	pass


class CamadaPadding(CStruct):
	# ${CamadaPadding}
	pass


class CamadaPool(CStruct):
	# ${CamadaPool}
	pass


class CamadaPoolAv(CStruct):
	# ${CamadaPoolAv}
	pass


class CamadaRelu(CStruct):
	# ${CamadaRelu}
	pass


class CamadaSoftMax(CStruct):
	# ${CamadaSoftMax}
	pass


class Estatistica(CStruct):
	# ${Estatistica}
	pass


class Cnn(CStruct):
	# ${Cnn}
	def __repr__(self):

		t = [f'CNN {c.addressof(self)} size {self.size}']
		try:
			for l in range(self.size):
				t.append(str(self.camadas[l]))
		except:
			pass
		return '\n'.join(t)

	def __init__(self, x, y, z, hitlearn=0.1, momento=0, decaimento=0):
		super(Cnn, self).__init__()
		self.__released__ = False
		clib.PY_createCnn(self.address(), hitlearn, momento, decaimento, x, y, z)

	def __del__(self):
		try:
			if self.__released__: return
			self.__released__ = True
			clib.PY_releaseCnn(self.address())
		except:
			pass

	def convolucao(self, passo, filtro, numeroFiltros, randomParam=[0, 0, 0]):
		if isinstance(passo, int): passo = [passo, passo]
		if isinstance(filtro, int): filtro = [filtro, filtro]
		erro = clib.Convolucao(self.address(), passo[0], passo[1], filtro[0], filtro[1], numeroFiltros, RandomParam(*randomParam))
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def convolucaoNc(self, passo, filtro, larg, numeroFiltros, randomParam=[0, 0, 0]):
		if isinstance(passo, int): passo = [passo, passo]
		if isinstance(filtro, int): filtro = [filtro, filtro]
		if isinstance(larg, int): larg = [larg, larg]
		erro = clib.ConvolucaoNcausal(self.address(), passo[0], passo[1], filtro[0], filtro[1], larg[0], larg[1], numeroFiltros, RandomParam(*randomParam))
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def Pooling(self, passo, filtro):
		if isinstance(passo, int): passo = [passo, passo]
		if isinstance(filtro, int): filtro = [filtro, filtro]
		erro = clib.Pooling(self.address(), passo[0], passo[1], filtro[0], filtro[1])
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def PoolingAv(self, passo, filtro):
		if isinstance(passo, int): passo = [passo, passo]
		if isinstance(filtro, int): filtro = [filtro, filtro]
		erro = clib.PoolingAv(self.address(), passo[0], passo[1], filtro[0], filtro[1])
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def Relu(self):
		erro = clib.Relu(self.address())
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def PRelu(self, PDF="default", a=0, b=0):
		PDF = PDF.lower()
		if PDF == "default":
			PDF = 0
		elif PDF == "normal":
			PDF = 1
		else:
			PDF = 2
		erro = clib.PRelu(self.address(), RandomParam(PDF, a, b))

		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def Padding(self, top, bottom, left, right):
		erro = clib.Padding(self.address(), top, bottom, left, right)
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def BatchNorm(self, epsilom=1e-13, randY=[0, 0, 0], randB=[0, 0, 0]):
		erro = clib.BatchNorm(self.address(), epsilom, RandomParam(*randY), RandomParam(*randB))
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def SoftMax(self, epsilom=1e-13):
		erro = clib.SoftMax(self.address())
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def Dropout(self, limiarSaida, seed=None):
		if not seed:
			seed = time.time()
		erro = clib.Dropout(self.address(), limiarSaida, int(seed))
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def FullConnect(self, saida, func_ativacao='TANH', randomParam=[0, 0, 0]):
		func_ativacao = func_ativacao.lower()
		if func_ativacao == 'tanh':
			func_ativacao = 2
		elif func_ativacao == 'sigmoid':
			func_ativacao = 0
		elif func_ativacao == 'relu':
			func_ativacao = 4
		else:
			raise Exception('Função de ativação invalida')
		erro = clib.FullConnect(self.address(), saida, func_ativacao, RandomParam(*randomParam))
		if erro: raise Exception(f"Falha ao adicionar camada:{inspect.stack()[0][3]}")
		return self.camadas[self.size - 1]

	def call(self, inputv):
		return clib.CnnCall(self.address(), self.toInput(inputv))

	def learn(self, target):
		return clib.CnnLearn(self.address(), self.toInput(target))

	def toInput(self, inputv):
		return (c.c_double * self.len_input)(*inputv)

	def toOutput(self, ot):
		return (c.c_double * self.len_input)(*ot)

	def save(self, file: str):
		file = file.encode('utf-8')
		return clib.CnnSaveInFile(self.address(), file)

	@staticmethod
	def load(file):
		cnn = Cnn(0, 0, 0)
		file = file.encode('utf-8')
		erro = clib.CnnLoadByFile(cnn.address(), file)
		if (erro): raise Exception(f'Error code {erro}')
		return cnn


class ManageTrain(CStruct):
	# ${ManageTrain}
	def chose2WorkDir(self):
		clib.manage2WorkDir(c.addressof(self))

	def __del__(self):
		if self.release: return
		self.release = True
		clib.releaseManageTrain(c.addressof(self))

	def setEvent(self, self_event, event):
		clib.manageTrainSetEvent(c.addressof(self_event), event)

	def setRun(self, can_run):
		can_run = int(can_run)
		clib.manageTrainSetRun(c.addressof(self), can_run)

	def __init__(self, luafile, taxa_aprendizado=0.1, momento=0, decaimento_peso=0,luaisFile=True):
		super().__init__()
		self.release = False
		if luaisFile:
			clib.createManageTrainPy(c.addressof(self), luafile.encode('utf-8'), taxa_aprendizado, momento, decaimento_peso)
		else:
			clib.createManageTrainPyStr(c.addressof(self), luafile.encode('utf-8'), taxa_aprendizado, momento, decaimento_peso)
		clib.manage2WorkDir(self.address())

	def loadImageStart(self, runBackground=True):
		clib.ManageTrainloadImages(self.address(), int(runBackground))

	def trainStart(self):
		clib.ManageTraintrain(c.addressof(self))

	def trainStart(self, runBackGround=True):
		clib.ManageTraintrain(self.address(), int(runBackGround))

	def startLoop(self, anotherThread=False):
		anotherThread = int(anotherThread)
		clib.manageTrainLoop(c.addressof(self), anotherThread)

	def save(self, file_name):
		file_name = file_name.encode('utf-8')
		return clib.CnnSaveInFile(c.addressof(self), file_name)


def SetSeed(seed):
	clib.initRandom(int(seed))


EVENT = c.CFUNCTYPE(None, TOPOINTER(ManageTrain))
Manage_p = TOPOINTER(ManageTrain)
