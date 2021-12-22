import ctypes

class RDP(ctypes.Structure):
	_fields_ = [('typpe', ctypes.c_int32), ('a', ctypes.c_float), ('b', ctypes.c_float)]

	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

class Params(ctypes.Structure):
	_fields_ = [('hitlearn', ctypes.c_float), ('momento', ctypes.c_float), ('decaimento', ctypes.c_float), ('learnable', ctypes.c_int32) ]	
	
	def __init__(self,  *args, **kw):
		super().__init__(*args, **kw)
		

class P2D(ctypes.Structure):
	_fields_ = [('x', ctypes.c_uint64), ('y', ctypes.c_uint64)]
	
	def __init__(self,  *args, **kw):
		super().__init__(*args, **kw)


class P3D(ctypes.Structure):
	_fields_ = [('x', ctypes.c_uint64), ('y', ctypes.c_uint64), ('z', ctypes.c_uint64)]
	
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)


class Cnn_t(ctypes.Structure):
	version: ctypes.c_char_p
	entrada: ctypes.c_void_p
	ds: ctypes.c_void_p
	target: ctypes.c_void_p
	l: ctypes.c_uint64
	cm: ctypes.c_void_p
	kernels: ctypes.c_void_p
	LuaVm: ctypes.c_void_p
	size_in: P3D
	erro: ctypes.c_void_p
	gpu: ctypes.c_void_p
	release_gpu: ctypes.c_int8
	queue: ctypes.c_void_p
	cdll_releaseL: ctypes.CFUNCTYPE(None, ctypes.c_void_p)
	cdll_setInput: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64)
	cdll_release: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)
	cdll_getSizeOut: ctypes.CFUNCTYPE(P3D, ctypes.c_void_p)
	cdll_json: ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.c_void_p,ctypes.c_int32)
	cdll_jsonF: ctypes.CFUNCTYPE(None, ctypes.c_void_p,ctypes.c_int32,ctypes.c_char_p)
	cdll_save: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_char_p)
	cdll_load: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_char_p)
	cdll_predict: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)
	cdll_predictv: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)
	cdll_learn: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)
	cdll_learnv: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)
	cdll_mse: ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p)
	cdll_mseT: ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p,ctypes.c_void_p)
	cdll_maxIndex: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)
	cdll_print: ctypes.CFUNCTYPE(None, ctypes.c_void_p,ctypes.c_char_p)
	cdll_printstr: ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.c_void_p,ctypes.c_char_p)
	cdll_normalizeIMAGE: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)
	cdll_extractVectorLabelClass: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)
	cdll_Convolucao: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P3D,Params,RDP)
	cdll_ConvolucaoF: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P3D,ctypes.c_void_p,Params,RDP)
	cdll_ConvolucaoNC: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P2D,P3D,ctypes.c_void_p,Params,RDP)
	cdll_Pooling: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P2D,ctypes.c_void_p)
	cdll_Relu: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_float,ctypes.c_float)
	cdll_PRelu: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,Params,RDP)
	cdll_FullConnect: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_uint64,Params,ctypes.c_void_p,RDP,RDP)
	cdll_Padding: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)
	cdll_DropOut: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_float,ctypes.c_void_p)
	cdll_SoftMax: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_int8)
	cdll_BatchNorm: ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_float,Params,RDP,RDP)
	cdll_removeLastLayer: ctypes.CFUNCTYPE(None, ctypes.c_void_p)
	_fields_ = [('version', ctypes.c_char_p),
			('entrada', ctypes.c_void_p),
			('ds', ctypes.c_void_p),
			('target', ctypes.c_void_p),
			('l', ctypes.c_uint64),
			('cm', ctypes.c_void_p),
			('kernels', ctypes.c_void_p),
			('LuaVm', ctypes.c_void_p),
			('size_in', P3D),
			('erro', ctypes.c_void_p),
			('gpu', ctypes.c_void_p),
			('release_gpu', ctypes.c_int8),
			('queue', ctypes.c_void_p),
			('cdll_releaseL', ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
			('cdll_setInput', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64)),
			('cdll_release', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)),
			('cdll_getSizeOut', ctypes.CFUNCTYPE(P3D, ctypes.c_void_p)),
			('cdll_json', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.c_void_p,ctypes.c_int32)),
			('cdll_jsonF', ctypes.CFUNCTYPE(None, ctypes.c_void_p,ctypes.c_int32,ctypes.c_char_p)),
			('cdll_save', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_char_p)),
			('cdll_load', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_char_p)),
			('cdll_predict', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_predictv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_learn', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_learnv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_mse', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p)),
			('cdll_mseT', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_maxIndex', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)),
			('cdll_print', ctypes.CFUNCTYPE(None, ctypes.c_void_p,ctypes.c_char_p)),
			('cdll_printstr', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.c_void_p,ctypes.c_char_p)),
			('cdll_normalizeIMAGE', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_extractVectorLabelClass', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_Convolucao', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P3D,Params,RDP)),
			('cdll_ConvolucaoF', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P3D,ctypes.c_void_p,Params,RDP)),
			('cdll_ConvolucaoNC', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P2D,P3D,ctypes.c_void_p,Params,RDP)),
			('cdll_Pooling', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,P2D,P2D,ctypes.c_void_p)),
			('cdll_Relu', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_float,ctypes.c_float)),
			('cdll_PRelu', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,Params,RDP)),
			('cdll_FullConnect', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_uint64,Params,ctypes.c_void_p,RDP,RDP)),
			('cdll_Padding', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)),
			('cdll_DropOut', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_float,ctypes.c_void_p)),
			('cdll_SoftMax', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_int8)),
			('cdll_BatchNorm', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,ctypes.c_float,Params,RDP,RDP)),
			('cdll_removeLastLayer', ctypes.CFUNCTYPE(None, ctypes.c_void_p))]
	@property
	def selfp(self):
		return  ctypes.cast(ctypes.addressof(self),ctypes.POINTER(Cnn_t))

	@property
	def selfpp(self):
		return ctypes.addressof(self.selfp)

gab_dll = ctypes.CDLL(r"C:\Users\hslhe\CLionProjects\Rede-convolucional\Gab\bin\libgab_library_cnn.dll")

gab_dll.Cnn_new.restype = ctypes.POINTER(ctypes.c_void_p)

gab_dll.gab_realloc.restype = ctypes.POINTER(ctypes.c_void_p)
gab_dll.gab_realloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint64]
class Cnn(ctypes.c_void_p):

	version: ctypes.c_char_p
	entrada: ctypes.c_void_p
	ds: ctypes.c_void_p
	target: ctypes.c_void_p
	l: ctypes.c_uint64
	cm: ctypes.c_void_p
	kernels: ctypes.c_void_p
	LuaVm: ctypes.c_void_p
	size_in: P3D
	erro: ctypes.c_void_p
	gpu: ctypes.c_void_p
	release_gpu: ctypes.c_int8
	queue: ctypes.c_void_p

	def __init__(self, *args, **kw):
		p = gab_dll.Cnn_new()
		ctypes.memmove(ctypes.addressof(self), ctypes.addressof(p), 8)
		self.py_reference = ctypes.cast(p, ctypes.POINTER(Cnn_t))

	def __getattribute__(self, item):
		try:
			return object.__getattribute__(self, item)
		except:
			return object.__getattribute__(self.py_reference[0], item)

	def __getitem__(self, item):
		try:
			return object.__getitem__(self, item)
		except:
			return object.__getitem__(self.py_reference[0], item)

	def __del__(self):
		self.cdll_release(self)
		pass
	

	def releaseL(self, L) -> None:
		rt_value = self.cdll_releaseL(L)
		return rt_value

	def setInput(self, x, y, z) -> ctypes.c_int32:
		rt_value = self.cdll_setInput(self, x, y, z)
		return rt_value

	def release(self, ) -> ctypes.c_int32:
		rt_value = self.cdll_release(self)
		return rt_value

	def getSizeOut(self, ) -> P3D:
		rt_value = self.cdll_getSizeOut(self)
		return rt_value

	def json(self, showValue) -> ctypes.POINTER(ctypes.c_char):
		rt_value = self.cdll_json(self, showValue)
		v = rt_value
		rt_value = ctypes.cast(v,ctypes.c_char_p).value.decode('utf-8')
		gab_dll.gab_free(v)
		return rt_value

	def jsonF(self, showValue, fileName) -> None:
		rt_value = self.cdll_jsonF(self, showValue, fileName)
		return rt_value

	def save(self, filename) -> ctypes.c_int32:
		rt_value = self.cdll_save(self, filename)
		return rt_value

	def load(self, filename) -> ctypes.c_int32:
		rt_value = self.cdll_load(self, filename)
		return rt_value

	def predict(self, input) -> ctypes.c_int32:
		rt_value = self.cdll_predict(self, input)
		return rt_value

	def predictv(self, input) -> ctypes.c_int32:
		rt_value = self.cdll_predictv(self, input)
		return rt_value

	def learn(self, target) -> ctypes.c_int32:
		rt_value = self.cdll_learn(self, target)
		return rt_value

	def learnv(self, target) -> ctypes.c_int32:
		rt_value = self.cdll_learnv(self, target)
		return rt_value

	def mse(self, ) -> ctypes.c_float:
		rt_value = self.cdll_mse(self)
		return rt_value

	def mseT(self, target) -> ctypes.c_float:
		rt_value = self.cdll_mseT(self, target)
		return rt_value

	def maxIndex(self, ) -> ctypes.c_int32:
		rt_value = self.cdll_maxIndex(self)
		return rt_value

	def print(self, comment) -> None:
		rt_value = self.cdll_print(self, comment)
		return rt_value

	def printstr(self, comment) -> ctypes.POINTER(ctypes.c_char):
		rt_value = self.cdll_printstr(self, comment)
		v = rt_value
		rt_value = ctypes.cast(v,ctypes.c_char_p).value.decode('utf-8')
		gab_dll.gab_free(v)
		return rt_value

	def normalizeIMAGE(self, dst_real, src_char) -> ctypes.c_int32:
		rt_value = self.cdll_normalizeIMAGE(self, dst_real, src_char)
		return rt_value

	def extractVectorLabelClass(self, dst, label) -> ctypes.c_int32:
		rt_value = self.cdll_extractVectorLabelClass(self, dst, label)
		return rt_value

	def Convolucao(self, passo, filtro, p, filtros) -> ctypes.c_int32:
		rt_value = self.cdll_Convolucao(self, passo, filtro, p, filtros)
		return rt_value

	def ConvolucaoF(self, passo, filtro, funcaoAtivacao, p, filtros) -> ctypes.c_int32:
		rt_value = self.cdll_ConvolucaoF(self, passo, filtro, funcaoAtivacao, p, filtros)
		return rt_value

	def ConvolucaoNC(self, passo, abertura, filtro, funcaoAtivacao, p, filtros) -> ctypes.c_int32:
		rt_value = self.cdll_ConvolucaoNC(self, passo, abertura, filtro, funcaoAtivacao, p, filtros)
		return rt_value

	def Pooling(self, passo, filtro, type) -> ctypes.c_int32:
		rt_value = self.cdll_Pooling(self, passo, filtro, type)
		return rt_value

	def Relu(self, fator_menor0, fator_maior0) -> ctypes.c_int32:
		rt_value = self.cdll_Relu(self, fator_menor0, fator_maior0)
		return rt_value

	def PRelu(self, params, rdp_a) -> ctypes.c_int32:
		rt_value = self.cdll_PRelu(self, params, rdp_a)
		return rt_value

	def FullConnect(self, numero_neuronios, p, funcaoAtivacao, rdp_pesos, rdp_bias) -> ctypes.c_int32:
		rt_value = self.cdll_FullConnect(self, numero_neuronios, p, funcaoAtivacao, rdp_pesos, rdp_bias)
		return rt_value

	def Padding(self, top, bottom, left, right) -> ctypes.c_int32:
		rt_value = self.cdll_Padding(self, top, bottom, left, right)
		return rt_value

	def DropOut(self, probabilidadeSaida, seed) -> ctypes.c_int32:
		rt_value = self.cdll_DropOut(self, probabilidadeSaida, seed)
		return rt_value

	def SoftMax(self, flag) -> ctypes.c_int32:
		rt_value = self.cdll_SoftMax(self, flag)
		return rt_value

	def BatchNorm(self, epsilon, p, randY, randB) -> ctypes.c_int32:
		rt_value = self.cdll_BatchNorm(self, epsilon, p, randY, randB)
		return rt_value

	def removeLastLayer(self, ) -> None:
		rt_value = self.cdll_removeLastLayer(self)
		return rt_value

a = Cnn()
a.print(None)