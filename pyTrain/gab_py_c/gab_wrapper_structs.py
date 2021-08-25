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


from gab_wrapper_load_dll import *
import numpy as np


class CStruct(c.Structure):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

	def __repr__(self):
		return str(type(self))

	def address(self):
		P = c.cast(c.addressof(self), c.POINTER(self.__class__))
		P.size = c.sizeof(self.__class__)
		return P

class Pointer(CStruct):
	p: c.c_void_p
	_fields_ = [
		('p', c.c_void_p),
	]

	pass


class Tensor(CStruct):
	data: c.c_void_p
	bytes: c.c_uint
	x: c.c_uint
	y: c.c_uint
	z: c.c_uint
	w: c.c_uint
	host: c.c_void_p
	flag: c.c_uint8
	context: c.c_void_p
	_fields_ = [
		('data', c.c_void_p),
		('bytes', c.c_uint),
		('x', c.c_uint),
		('y', c.c_uint),
		('z', c.c_uint),
		('w', c.c_uint),
		('host', c.c_void_p),
		('flag', c.c_uint8),
		('context', c.c_void_p),
	]

	def getvalues(self, ):
		pass

	def getvalues_np(self):
		pass


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


class List_args(CStruct):
	values: c.POINTER(Dbchar_p)
	size: c.c_int
	_fields_ = [
		('values', c.POINTER(Dbchar_p)),
		('size', c.c_int),
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
	msg: c.c_char_p * EXCEPTION_MAX_MSG_SIZE
	_fields_ = [
		('error', c.c_int),
		('msg', c.c_char_p * EXCEPTION_MAX_MSG_SIZE),
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
	flag_notlearn: c.c_char
	flag_usehost: c.c_char
	parametros: Params
	gradsEntrada: TOPOINTER(Tensor)
	entrada: TOPOINTER(Tensor)
	saida: TOPOINTER(Tensor)
	queue: c.c_void_p
	context: c.c_void_p
	max_works: c.POINTER(c.c_size_t)
	calc_grads: c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)
	corrige_pesos: c.CFUNCTYPE(c.c_int, c.c_void_p)
	ativa: c.CFUNCTYPE(c.c_int, c.c_void_p)
	release: c.CFUNCTYPE(c.c_int, c.c_void_p)
	salvar: c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p)
	toString: c.CFUNCTYPE(c.c_char_p, c.c_void_p)
	getCreateParams: c.CFUNCTYPE(c.c_char_p, c.c_void_p)
	setLearn: c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_char)
	setParams: c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_double, c.c_double, c.c_double)
	__string__: c.c_char_p
	_fields_ = [
		('type', c.c_char),
		('flag_releaseInput', c.c_char),
		('flag_notlearn', c.c_char),
		('flag_usehost', c.c_char),
		('parametros', Params),
		('gradsEntrada', TOPOINTER(Tensor)),
		('entrada', TOPOINTER(Tensor)),
		('saida', TOPOINTER(Tensor)),
		('queue', c.c_void_p),
		('context', c.c_void_p),
		('max_works', c.POINTER(c.c_size_t)),
		('calc_grads', c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)),
		('corrige_pesos', c.CFUNCTYPE(c.c_int, c.c_void_p)),
		('ativa', c.CFUNCTYPE(c.c_int, c.c_void_p)),
		('release', c.CFUNCTYPE(c.c_int, c.c_void_p)),
		('salvar', c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p)),
		('toString', c.CFUNCTYPE(c.c_char_p, c.c_void_p)),
		('getCreateParams', c.CFUNCTYPE(c.c_char_p, c.c_void_p)),
		('setLearn', c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_char)),
		('setParams', c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_double, c.c_double, c.c_double)),
		('__string__', c.c_char_p),
	]

	def __repr__(self):
		t = self.toString(c.addressof(self))
		t = t.value.decode('utf-8')
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
	kernelSoftMaxCalcGrads: c.c_void_p
	soma: TOPOINTER(Tensor)
	exponent: TOPOINTER(Tensor)
	_fields_ = [
		('super', Camada),
		('kernelSoftMaxAtiva1', c.c_void_p),
		('kernelSoftMaxAtiva2', c.c_void_p),
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
	]

	pass


class Cnn(CStruct):
	parametros: Params
	camadas: c.POINTER(Camada)
	lastGrad: TOPOINTER(Tensor)
	target: TOPOINTER(Tensor)
	size: c.c_int
	sizeIn: Ponto
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
	luaArgs: List_args
	releaseL: c.CFUNCTYPE(c.c_int, c.c_void_p)
	error: CNN_ERROR
	release_self: c.c_char
	_fields_ = [
		('parametros', Params),
		('camadas', c.POINTER(Camada)),
		('lastGrad', TOPOINTER(Tensor)),
		('target', TOPOINTER(Tensor)),
		('size', c.c_int),
		('sizeIn', Ponto),
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
		('luaArgs', List_args),
		('releaseL', c.CFUNCTYPE(c.c_int, c.c_void_p)),
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
		print('init', c.sizeof(Cnn))
		clib.createCnnPy(self.address(), hitlearn, momento, decaimento, x, y, z)
	def address(self):
		P = c.cast(c.addressof(self), c.POINTER(Cnn))
		P.size = c.sizeof(self.__class__)
		return P
	def __del__(self):
		# clib.releaseCnnWrapper(self.address())
		pass


class ManageTrain(CStruct):
	et: Estatistica
	cnn: Cnn
	homePath: String
	file_images: String
	file_labels: String
	headers_images: c.c_uint
	headers_labels: c.c_uint
	imagens: TOPOINTER(Tensor)
	targets: TOPOINTER(Tensor)
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
	OnloadedImages: c.CFUNCTYPE(None, c.c_void_p)
	OnfinishEpic: c.CFUNCTYPE(None, c.c_void_p)
	OnInitTrain: c.CFUNCTYPE(None, c.c_void_p)
	OnfinishTrain: c.CFUNCTYPE(None, c.c_void_p)
	OnInitFitnes: c.CFUNCTYPE(None, c.c_void_p)
	OnfinishFitnes: c.CFUNCTYPE(None, c.c_void_p)
	UpdateTrain: c.CFUNCTYPE(None, c.c_void_p)
	UpdateFitnes: c.CFUNCTYPE(None, c.c_void_p)
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
		('imagens', TOPOINTER(Tensor)),
		('targets', TOPOINTER(Tensor)),
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
		('OnloadedImages', c.CFUNCTYPE(None, c.c_void_p)),
		('OnfinishEpic', c.CFUNCTYPE(None, c.c_void_p)),
		('OnInitTrain', c.CFUNCTYPE(None, c.c_void_p)),
		('OnfinishTrain', c.CFUNCTYPE(None, c.c_void_p)),
		('OnInitFitnes', c.CFUNCTYPE(None, c.c_void_p)),
		('OnfinishFitnes', c.CFUNCTYPE(None, c.c_void_p)),
		('UpdateTrain', c.CFUNCTYPE(None, c.c_void_p)),
		('UpdateFitnes', c.CFUNCTYPE(None, c.c_void_p)),
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
