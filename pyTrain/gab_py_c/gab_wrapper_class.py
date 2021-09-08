from gab_wrapper_load_dll import *
import numpy as np


class CStruct(c.Structure):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

	def __repr__(self):
		return str(type(self))

	def address(self):
		P = c.cast(c.addressof(self), c.POINTER(Cnn))
		P.size = c.sizeof(self.__class__)
		return P


class Pointer(CStruct):
	# ${Pointer}
	pass


class Tensor(CStruct):
	# ${Tensor}
	def getvalues(self, ):
		pass

	def getvalues_np(self):
		pass


class String(CStruct):
	# ${String}
	pass


class Dbchar_p(CStruct):
	# ${Dbchar_p}
	pass


class List_args(CStruct):
	# ${List_args}
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
		t = t.value.decode('utf-8')
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
		self.__released__= False
		clib.PY_createCnn(self.address(), hitlearn, momento, decaimento, x, y, z)

	def __del__(self):
		if self.__released__: return
		self.__released__ = True
		clib.PY_releaseCnn(self.address())
	def convolucao(self,passo,filtro,numeroFiltros):
		if not isinstance(passo,(list,tuple)):

		clib.Convolucao()

	int Convolucao(Cnn c, char tensor_flag, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT numeroDeFiltros);

	int ConvolucaoNcausal(Cnn c, char tensor_flag, UINT passox, UINT passoy, UINT filtrox, UINT filtroy,
																								UINT largx, UINT largy,
																												 UINT numeroDeFiltros);

	int Pooling(Cnn c, char tensor_flag, UINT passox, UINT passoy,
														   UINT filtrox, UINT filtroy);

	int PoolingAv(Cnn c, char tensor_flag, UINT passox, UINT pasoy, UINT fx, UINT fy);

	int Relu(Cnn c, char tensor_flag);

	int Padding(Cnn c, char tensor_flag, UINT top, UINT bottom, UINT left, UINT right);

	int BatchNorm(Cnn c, char tensor_flag, double epsilon);

	int SoftMax(Cnn c, char tensor_flag);

	int Dropout(Cnn c, char tensor_flag, double pontoAtivacao, long long int seed);

	int FullConnect(Cnn c, char tensor_flag, UINT tamanhoDaSaida, int funcaoDeAtivacao);


	class ManageTrain(CStruct):
	# ${ManageTrain}
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
