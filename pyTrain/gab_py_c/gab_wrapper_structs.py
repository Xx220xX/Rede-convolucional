import ctypes as c

EXCEPTION_MAX_MSG_SIZE = 500


def TOPOINTER(c_type):
	tp = c.POINTER(c_type)

	def get(self, item):
		return self[0].__getattribute__(item)

	def set(self, key, value):
		self[0].__setattr__(key, value)

	def rep(self):
		return self[0].__repr__()

	tp.__getattribute__ = get
	tp.__setattr__ = set
	tp.__repr__ = rep
	return tp
    
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
		P = c.cast(c.addressof(self), c.POINTER(Cnn))
		P.size = c.sizeof(self.__class__)
		return P


class Pointer(CStruct):
	p: c.c_void_p
	_fields_ = [
		('p', c.c_void_p),
	]
	
	pass


class RandomParam(CStruct):
	type: c.c_int
	a: c.c_double
	b: c.c_double
	_fields_ = [
		('type', c.c_int),
		('a', c.c_double),
		('b', c.c_double),
	]
	
	pass


class Tensor(CStruct):
	data: c.c_void_p
	bytes: c.c_uint
	x: c.c_uint
	y: c.c_uint
	z: c.c_uint
	w: c.c_uint
	var_host: c.c_void_p
	flag: c.c_uint8
	context: c.c_void_p
	_fields_ = [
		('data', c.c_void_p),
		('bytes', c.c_uint),
		('x', c.c_uint),
		('y', c.c_uint),
		('z', c.c_uint),
		('w', c.c_uint),
		('var_host', c.c_void_p),
		('flag', c.c_uint8),
		('context', c.c_void_p),
	]
	
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
	d: c.c_char_p
	size: c.c_size_t
	release: c.c_char
	_fields_ = [
		('d', c.c_char_p),
		('size', c.c_size_t),
		('release', c.c_char),
	]
	
	pass


class Dbchar_p(CStruct):
	name: c.c_char_p
	value: c.c_char_p
	_fields_ = [
		('name', c.c_char_p),
		('value', c.c_char_p),
	]
	
	pass


class Dictionary(CStruct):
	values: c.POINTER(Dbchar_p)
	size: c.c_int
	self_release: c.c_char
	_fields_ = [
		('values', c.POINTER(Dbchar_p)),
		('size', c.c_int),
		('self_release', c.c_char),
	]
	
	pass


class Ponto(CStruct):
	x: c.c_size_t
	y: c.c_size_t
	z: c.c_size_t
	_fields_ = [
		('x', c.c_size_t),
		('y', c.c_size_t),
		('z', c.c_size_t),
	]
	
	pass


class Kernel(CStruct):
	kernel: c.c_void_p
	kernel_name: c.c_char_p
	nArgs: c.c_int
	l_args: c.POINTER(c.c_size_t)
	_fields_ = [
		('kernel', c.c_void_p),
		('kernel_name', c.c_char_p),
		('nArgs', c.c_int),
		('l_args', c.POINTER(c.c_size_t)),
	]
	
	pass


class CNN_ERROR(CStruct):
	error: c.c_int
	msg: c.c_char_p*EXCEPTION_MAX_MSG_SIZE
	_fields_ = [
		('error', c.c_int),
		('msg', c.c_char_p*EXCEPTION_MAX_MSG_SIZE),
	]
	
	pass


class Params(CStruct):
	hitLearn: c.c_double
	momento: c.c_double
	decaimentoDePeso: c.c_double
	_fields_ = [
		('hitLearn', c.c_double),
		('momento', c.c_double),
		('decaimentoDePeso', c.c_double),
	]
	
	pass


class Camada(CStruct):
	type: c.c_char
	flag_releaseInput: c.c_char
	learnable: c.c_char
	parametros: Params
	gradsEntrada: TOPOINTER(Tensor)
	entrada: TOPOINTER(Tensor)
	saida: TOPOINTER(Tensor)
	queue: c.c_void_p
	context: c.c_void_p
	max_works: c.POINTER(c.c_size_t)
	backpropagation: c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_void_p)
	propagation: c.CFUNCTYPE(c.c_int,c.c_void_p)
	release: c.CFUNCTYPE(c.c_int,c.c_void_p)
	salvar: c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_void_p,c.c_void_p,c.c_void_p)
	toString: c.CFUNCTYPE(c.c_char_p,c.c_void_p)
	getCreateParams: c.CFUNCTYPE(c.c_char_p,c.c_void_p)
	setLearn: c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_char)
	setParams: c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_double,c.c_double,c.c_double)
	__string__: c.c_char_p
	_fields_ = [
		('type', c.c_char),
		('flag_releaseInput', c.c_char),
		('learnable', c.c_char),
		('parametros', Params),
		('gradsEntrada', TOPOINTER(Tensor)),
		('entrada', TOPOINTER(Tensor)),
		('saida', TOPOINTER(Tensor)),
		('queue', c.c_void_p),
		('context', c.c_void_p),
		('max_works', c.POINTER(c.c_size_t)),
		('backpropagation', c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_void_p)),
		('propagation', c.CFUNCTYPE(c.c_int,c.c_void_p)),
		('release', c.CFUNCTYPE(c.c_int,c.c_void_p)),
		('salvar', c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_void_p,c.c_void_p,c.c_void_p)),
		('toString', c.CFUNCTYPE(c.c_char_p,c.c_void_p)),
		('getCreateParams', c.CFUNCTYPE(c.c_char_p,c.c_void_p)),
		('setLearn', c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_char)),
		('setParams', c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_double,c.c_double,c.c_double)),
		('__string__', c.c_char_p),
	]
	
	def __repr__(self):
		t = self.toString(c.addressof(self))
		t = t.decode('utf-8')
		return t


class CamadaBatchNorm(CStruct):
	super: Camada
	Y: TOPOINTER(Tensor)
	gradY: TOPOINTER(Tensor)
	B: TOPOINTER(Tensor)
	gradB: TOPOINTER(Tensor)
	epsilon: c.c_double
	media: TOPOINTER(Tensor)
	somaDiferenca: TOPOINTER(Tensor)
	variancia: TOPOINTER(Tensor)
	gradVariancia: TOPOINTER(Tensor)
	diferenca: TOPOINTER(Tensor)
	diferencaquad: TOPOINTER(Tensor)
	norma: TOPOINTER(Tensor)
	kernelBatchNormAtiva1: c.c_void_p
	kernelBatchNormAtiva2: c.c_void_p
	kernelBatchNormAtiva3: c.c_void_p
	kernelBatchNormAtiva4: c.c_void_p
	kernelBatchNormCalcGrads1: c.c_void_p
	kernelBatchNormCalcGrads2: c.c_void_p
	kernelBatchNormCorrige: c.c_void_p
	_fields_ = [
		('super', Camada),
		('Y', TOPOINTER(Tensor)),
		('gradY', TOPOINTER(Tensor)),
		('B', TOPOINTER(Tensor)),
		('gradB', TOPOINTER(Tensor)),
		('epsilon', c.c_double),
		('media', TOPOINTER(Tensor)),
		('somaDiferenca', TOPOINTER(Tensor)),
		('variancia', TOPOINTER(Tensor)),
		('gradVariancia', TOPOINTER(Tensor)),
		('diferenca', TOPOINTER(Tensor)),
		('diferencaquad', TOPOINTER(Tensor)),
		('norma', TOPOINTER(Tensor)),
		('kernelBatchNormAtiva1', c.c_void_p),
		('kernelBatchNormAtiva2', c.c_void_p),
		('kernelBatchNormAtiva3', c.c_void_p),
		('kernelBatchNormAtiva4', c.c_void_p),
		('kernelBatchNormCalcGrads1', c.c_void_p),
		('kernelBatchNormCalcGrads2', c.c_void_p),
		('kernelBatchNormCorrige', c.c_void_p),
	]
	
	pass


class CamadaConv(CStruct):
	super: Camada
	filtros: TOPOINTER(Tensor)
	grad_filtros: TOPOINTER(Tensor)
	gradnext: TOPOINTER(Tensor)
	passox: c.c_uint
	passoy: c.c_uint
	kernelConvSum: c.c_void_p
	kernelConvFixWeight: c.c_void_p
	kernelConvCalcGrads: c.c_void_p
	_fields_ = [
		('super', Camada),
		('filtros', TOPOINTER(Tensor)),
		('grad_filtros', TOPOINTER(Tensor)),
		('gradnext', TOPOINTER(Tensor)),
		('passox', c.c_uint),
		('passoy', c.c_uint),
		('kernelConvSum', c.c_void_p),
		('kernelConvFixWeight', c.c_void_p),
		('kernelConvCalcGrads', c.c_void_p),
	]
	
	pass


class CamadaConvNc(CStruct):
	super: Camada
	filtros: TOPOINTER(Tensor)
	grad_filtros: TOPOINTER(Tensor)
	grad_filtros_old: TOPOINTER(Tensor)
	passox: c.c_uint
	passoy: c.c_uint
	largx: c.c_uint
	largy: c.c_uint
	numeroFiltros: c.c_uint
	kernelConvNcSum: c.c_void_p
	kernelConvNcFixWeight: c.c_void_p
	kernelConvNcCalcGradsFiltro: c.c_void_p
	kernelConvNcCalcGrads: c.c_void_p
	_fields_ = [
		('super', Camada),
		('filtros', TOPOINTER(Tensor)),
		('grad_filtros', TOPOINTER(Tensor)),
		('grad_filtros_old', TOPOINTER(Tensor)),
		('passox', c.c_uint),
		('passoy', c.c_uint),
		('largx', c.c_uint),
		('largy', c.c_uint),
		('numeroFiltros', c.c_uint),
		('kernelConvNcSum', c.c_void_p),
		('kernelConvNcFixWeight', c.c_void_p),
		('kernelConvNcCalcGradsFiltro', c.c_void_p),
		('kernelConvNcCalcGrads', c.c_void_p),
	]
	
	pass


class CamadaDropOut(CStruct):
	super: Camada
	hitmap: TOPOINTER(Tensor)
	flag_releaseInput: c.c_char
	p_ativacao: c.c_double
	seed: c.c_uint64
	kerneldropativa: c.c_void_p
	kerneldropcalcgrad: c.c_void_p
	_fields_ = [
		('super', Camada),
		('hitmap', TOPOINTER(Tensor)),
		('flag_releaseInput', c.c_char),
		('p_ativacao', c.c_double),
		('seed', c.c_uint64),
		('kerneldropativa', c.c_void_p),
		('kerneldropcalcgrad', c.c_void_p),
	]
	
	pass


class CamadaFullConnect(CStruct):
	super: Camada
	pesos: TOPOINTER(Tensor)
	grad: TOPOINTER(Tensor)
	z: TOPOINTER(Tensor)
	dz: TOPOINTER(Tensor)
	fa: c.c_int
	dfa: c.c_int
	kernelfullfeed: c.c_void_p
	kernelfullfixWeight: c.c_void_p
	kernelfullcalcgrad1: c.c_void_p
	kernelfullcalcgrad2: c.c_void_p
	_fields_ = [
		('super', Camada),
		('pesos', TOPOINTER(Tensor)),
		('grad', TOPOINTER(Tensor)),
		('z', TOPOINTER(Tensor)),
		('dz', TOPOINTER(Tensor)),
		('fa', c.c_int),
		('dfa', c.c_int),
		('kernelfullfeed', c.c_void_p),
		('kernelfullfixWeight', c.c_void_p),
		('kernelfullcalcgrad1', c.c_void_p),
		('kernelfullcalcgrad2', c.c_void_p),
	]
	
	pass


class CamadaPadding(CStruct):
	super: Camada
	top: c.c_size_t
	bottom: c.c_size_t
	left: c.c_size_t
	right: c.c_size_t
	ativa: c.c_void_p
	calcGrad: c.c_void_p
	_fields_ = [
		('super', Camada),
		('top', c.c_size_t),
		('bottom', c.c_size_t),
		('left', c.c_size_t),
		('right', c.c_size_t),
		('ativa', c.c_void_p),
		('calcGrad', c.c_void_p),
	]
	
	pass


class CamadaPool(CStruct):
	super: Camada
	passox: c.c_uint
	passoy: c.c_uint
	filtrox: c.c_uint
	filtroy: c.c_uint
	kernelPoolAtiva: c.c_void_p
	kernelPoolCalcGrads: c.c_void_p
	_fields_ = [
		('super', Camada),
		('passox', c.c_uint),
		('passoy', c.c_uint),
		('filtrox', c.c_uint),
		('filtroy', c.c_uint),
		('kernelPoolAtiva', c.c_void_p),
		('kernelPoolCalcGrads', c.c_void_p),
	]
	
	pass


class CamadaPoolAv(CStruct):
	super: Camada
	passox: c.c_uint
	passoy: c.c_uint
	fx: c.c_uint
	fy: c.c_uint
	kernelPoolAvAtiva: c.c_void_p
	kernelPoolAvCalcGrads: c.c_void_p
	_fields_ = [
		('super', Camada),
		('passox', c.c_uint),
		('passoy', c.c_uint),
		('fx', c.c_uint),
		('fy', c.c_uint),
		('kernelPoolAvAtiva', c.c_void_p),
		('kernelPoolAvCalcGrads', c.c_void_p),
	]
	
	pass


class CamadaRelu(CStruct):
	super: Camada
	kernelReluAtiva: c.c_void_p
	kernelReluCalcGrads: c.c_void_p
	_fields_ = [
		('super', Camada),
		('kernelReluAtiva', c.c_void_p),
		('kernelReluCalcGrads', c.c_void_p),
	]
	
	pass


class CamadaSoftMax(CStruct):
	super: Camada
	kernelSoftMaxAtiva1: c.c_void_p
	kernelSoftMaxAtiva2: c.c_void_p
	kernelSoftMaxAtiva3: c.c_void_p
	kernelSoftMaxCalcGrads: c.c_void_p
	soma: TOPOINTER(Tensor)
	exponent: TOPOINTER(Tensor)
	_fields_ = [
		('super', Camada),
		('kernelSoftMaxAtiva1', c.c_void_p),
		('kernelSoftMaxAtiva2', c.c_void_p),
		('kernelSoftMaxAtiva3', c.c_void_p),
		('kernelSoftMaxCalcGrads', c.c_void_p),
		('soma', TOPOINTER(Tensor)),
		('exponent', TOPOINTER(Tensor)),
	]
	
	pass


class Estatistica(CStruct):
	tr_mse_vector: c.POINTER(c.c_double)
	tr_acertos_vector: c.POINTER(c.c_double)
	tr_imagem_atual: c.c_uint
	tr_numero_imagens: c.c_uint
	tr_epoca_atual: c.c_uint
	tr_numero_epocas: c.c_uint
	tr_erro_medio: c.c_double
	tr_acerto_medio: c.c_double
	tr_time: c.c_size_t
	ft_imagem_atual: c.c_uint
	ft_numero_imagens: c.c_uint
	ft_info: c.POINTER(c.c_double)
	ft_numero_classes: c.c_uint
	ft_time: c.c_size_t
	ll_imagem_atual: c.c_size_t
	ld_imagem_atual: c.c_size_t
	_fields_ = [
		('tr_mse_vector', c.POINTER(c.c_double)),
		('tr_acertos_vector', c.POINTER(c.c_double)),
		('tr_imagem_atual', c.c_uint),
		('tr_numero_imagens', c.c_uint),
		('tr_epoca_atual', c.c_uint),
		('tr_numero_epocas', c.c_uint),
		('tr_erro_medio', c.c_double),
		('tr_acerto_medio', c.c_double),
		('tr_time', c.c_size_t),
		('ft_imagem_atual', c.c_uint),
		('ft_numero_imagens', c.c_uint),
		('ft_info', c.POINTER(c.c_double)),
		('ft_numero_classes', c.c_uint),
		('ft_time', c.c_size_t),
		('ll_imagem_atual', c.c_size_t),
		('ld_imagem_atual', c.c_size_t),
	]
	
	pass


class Cnn(CStruct):
	parametros: Params
	camadas: c.POINTER(TOPOINTER(Camada))
	lastGrad: TOPOINTER(Tensor)
	target: TOPOINTER(Tensor)
	size: c.c_int
	sizeIn: Ponto
	len_input: c.c_size_t
	len_output: c.c_size_t
	queue: c.c_void_p
	cl: c.c_void_p
	releaseCL: c.c_char
	kernelsub: c.c_void_p
	kerneldiv: c.c_void_p
	kerneldivInt: c.c_void_p
	kernelNormalize: c.c_void_p
	kernelInt2Vector: c.c_void_p
	kernelcreateIMG: c.c_void_p
	L: c.c_void_p
	luaArgs: Dictionary
	releaseL: c.CFUNCTYPE(c.c_int,c.c_void_p)
	error: CNN_ERROR
	release_self: c.c_char
	_fields_ = [
		('parametros', Params),
		('camadas', c.POINTER(TOPOINTER(Camada))),
		('lastGrad', TOPOINTER(Tensor)),
		('target', TOPOINTER(Tensor)),
		('size', c.c_int),
		('sizeIn', Ponto),
		('len_input', c.c_size_t),
		('len_output', c.c_size_t),
		('queue', c.c_void_p),
		('cl', c.c_void_p),
		('releaseCL', c.c_char),
		('kernelsub', c.c_void_p),
		('kerneldiv', c.c_void_p),
		('kerneldivInt', c.c_void_p),
		('kernelNormalize', c.c_void_p),
		('kernelInt2Vector', c.c_void_p),
		('kernelcreateIMG', c.c_void_p),
		('L', c.c_void_p),
		('luaArgs', Dictionary),
		('releaseL', c.CFUNCTYPE(c.c_int,c.c_void_p)),
		('error', CNN_ERROR),
		('release_self', c.c_char),
	]
	
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
		if self.__released__: return
		self.__released__ = True
		clib.PY_releaseCnn(self.address())

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
	et: Estatistica
	cnn: Cnn
	homePath: String
	file_images: String
	file_labels: String
	headers_images: c.c_uint
	headers_labels: c.c_uint
	imagens: c.POINTER(TOPOINTER(Tensor))
	targets: c.POINTER(TOPOINTER(Tensor))
	labels: TOPOINTER(Tensor)
	n_epics: c.c_int
	epic: c.c_int
	n_images: c.c_int
	n_images2train: c.c_int
	n_images2fitness: c.c_int
	image: c.c_int
	n_classes: c.c_int
	class_names: String
	character_sep: c.c_char
	sum_erro: c.c_double
	sum_acerto: c.c_int
	current_time: c.c_double
	OnloadedImages: c.CFUNCTYPE(None,c.c_void_p)
	OnfinishEpic: c.CFUNCTYPE(None,c.c_void_p)
	OnInitTrain: c.CFUNCTYPE(None,c.c_void_p)
	OnfinishTrain: c.CFUNCTYPE(None,c.c_void_p)
	OnInitFitnes: c.CFUNCTYPE(None,c.c_void_p)
	OnfinishFitnes: c.CFUNCTYPE(None,c.c_void_p)
	UpdateTrain: c.CFUNCTYPE(None,c.c_void_p)
	UpdateFitnes: c.CFUNCTYPE(None,c.c_void_p)
	UpdateLoad: c.CFUNCTYPE(None,c.c_void_p)
	self_release: c.c_char
	real_time: c.c_char
	process: c.c_void_p
	update_loop: c.c_void_p
	can_run: c.c_uint
	process_id: c.c_uint
	_fields_ = [
		('et', Estatistica),
		('cnn', Cnn),
		('homePath', String),
		('file_images', String),
		('file_labels', String),
		('headers_images', c.c_uint),
		('headers_labels', c.c_uint),
		('imagens', c.POINTER(TOPOINTER(Tensor))),
		('targets', c.POINTER(TOPOINTER(Tensor))),
		('labels', TOPOINTER(Tensor)),
		('n_epics', c.c_int),
		('epic', c.c_int),
		('n_images', c.c_int),
		('n_images2train', c.c_int),
		('n_images2fitness', c.c_int),
		('image', c.c_int),
		('n_classes', c.c_int),
		('class_names', String),
		('character_sep', c.c_char),
		('sum_erro', c.c_double),
		('sum_acerto', c.c_int),
		('current_time', c.c_double),
		('OnloadedImages', c.CFUNCTYPE(None,c.c_void_p)),
		('OnfinishEpic', c.CFUNCTYPE(None,c.c_void_p)),
		('OnInitTrain', c.CFUNCTYPE(None,c.c_void_p)),
		('OnfinishTrain', c.CFUNCTYPE(None,c.c_void_p)),
		('OnInitFitnes', c.CFUNCTYPE(None,c.c_void_p)),
		('OnfinishFitnes', c.CFUNCTYPE(None,c.c_void_p)),
		('UpdateTrain', c.CFUNCTYPE(None,c.c_void_p)),
		('UpdateFitnes', c.CFUNCTYPE(None,c.c_void_p)),
		('UpdateLoad', c.CFUNCTYPE(None,c.c_void_p)),
		('self_release', c.c_char),
		('real_time', c.c_char),
		('process', c.c_void_p),
		('update_loop', c.c_void_p),
		('can_run', c.c_uint),
		('process_id', c.c_uint),
	]
	
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


def SetSeed(seed):
	clib.initRandom(int(seed))
