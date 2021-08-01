from wrapper_dll import *

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


class Kernel(c.Structure):
	_fields_ = [
		('kernel', c.c_void_p),
		('kernel_name', c.c_char_p),
		('nArgs', c.c_int),
		('l_args', c.POINTER(c.c_size_t)),
	]


class Exception(c.Structure):
	_fields_ = [
		('error', c.c_int),
		('msg', c.c_char*EXCEPTION_MAX_MSG_SIZE),
	]


class Ponto(c.Structure):
	_fields_ = [
		('x', c.c_size_t),
		('y', c.c_size_t),
		('z', c.c_size_t),
	]


class Tensor(c.Structure):
	_fields_ = [
		('data', c.c_void_p),
		('bytes', c.c_void_p),
		('x', c.c_void_p),
		('y', c.c_void_p),
		('z', c.c_void_p),
		('w', c.c_void_p),
		('host', c.c_void_p),
		('flag', c.c_uint8),
		('context', c.c_void_p),
	]


class Params(c.Structure):
	_fields_ = [
		('hitLearn', c.c_double),
		('momento', c.c_double),
		('decaimentoDePeso', c.c_double),
	]


class Camada(c.Structure):
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


class CamadaBatchNorm(c.Structure):
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
		('kernelBatchNormAtiva1', TOPOINTER(Kernel)),
		('kernelBatchNormAtiva2', TOPOINTER(Kernel)),
		('kernelBatchNormAtiva3', TOPOINTER(Kernel)),
		('kernelBatchNormAtiva4', TOPOINTER(Kernel)),
		('kernelBatchNormCalcGrads1', TOPOINTER(Kernel)),
		('kernelBatchNormCalcGrads2', TOPOINTER(Kernel)),
		('kernelBatchNormCorrige', TOPOINTER(Kernel)),
	]


class CamadaConv(c.Structure):
	_fields_ = [
		('super', Camada),
		('filtros', TOPOINTER(Tensor)),
		('grad_filtros', TOPOINTER(Tensor)),
		('gradnext', TOPOINTER(Tensor)),
		('passox', c.c_void_p),
		('passoy', c.c_void_p),
		('kernelConvSum', TOPOINTER(Kernel)),
		('kernelConvFixWeight', TOPOINTER(Kernel)),
		('kernelConvCalcGrads', TOPOINTER(Kernel)),
	]


class CamadaConvNc(c.Structure):
	_fields_ = [
		('super', Camada),
		('filtros', TOPOINTER(Tensor)),
		('grad_filtros', TOPOINTER(Tensor)),
		('grad_filtros_old', TOPOINTER(Tensor)),
		('passox', c.c_void_p),
		('passoy', c.c_void_p),
		('largx', c.c_void_p),
		('largy', c.c_void_p),
		('numeroFiltros', c.c_void_p),
		('kernelConvNcSum', TOPOINTER(Kernel)),
		('kernelConvNcFixWeight', TOPOINTER(Kernel)),
		('kernelConvNcCalcGradsFiltro', TOPOINTER(Kernel)),
		('kernelConvNcCalcGrads', TOPOINTER(Kernel)),
	]


class CamadaDropOut(c.Structure):
	_fields_ = [
		('super', Camada),
		('hitmap', TOPOINTER(Tensor)),
		('flag_releaseInput', c.c_char),
		('p_ativacao', c.c_double),
		('seed', c.c_uint64),
		('kerneldropativa', TOPOINTER(Kernel)),
		('kerneldropcalcgrad', TOPOINTER(Kernel)),
	]


class CamadaFullConnect(c.Structure):
	_fields_ = [
		('super', Camada),
		('pesos', TOPOINTER(Tensor)),
		('grad', TOPOINTER(Tensor)),
		('z', TOPOINTER(Tensor)),
		('dz', TOPOINTER(Tensor)),
		('fa', c.c_int),
		('dfa', c.c_int),
		('kernelfullfeed', TOPOINTER(Kernel)),
		('kernelfullfixWeight', TOPOINTER(Kernel)),
		('kernelfullcalcgrad1', TOPOINTER(Kernel)),
		('kernelfullcalcgrad2', TOPOINTER(Kernel)),
	]


class CamadaPadding(c.Structure):
	_fields_ = [
		('super', Camada),
		('top', c.c_size_t),
		('bottom', c.c_size_t),
		('left', c.c_size_t),
		('right', c.c_size_t),
		('ativa', TOPOINTER(Kernel)),
		('calcGrad', TOPOINTER(Kernel)),
	]


class CamadaPool(c.Structure):
	_fields_ = [
		('super', Camada),
		('passox', c.c_void_p),
		('passoy', c.c_void_p),
		('filtrox', c.c_void_p),
		('filtroy', c.c_void_p),
		('kernelPoolAtiva', TOPOINTER(Kernel)),
		('kernelPoolCalcGrads', TOPOINTER(Kernel)),
	]


class CamadaPoolAv(c.Structure):
	_fields_ = [
		('super', Camada),
		('passox', c.c_void_p),
		('passoy', c.c_void_p),
		('fx', c.c_void_p),
		('fy', c.c_void_p),
		('kernelPoolAvAtiva', TOPOINTER(Kernel)),
		('kernelPoolAvCalcGrads', TOPOINTER(Kernel)),
	]


class CamadaRelu(c.Structure):
	_fields_ = [
		('super', Camada),
		('kernelReluAtiva', TOPOINTER(Kernel)),
		('kernelReluCalcGrads', TOPOINTER(Kernel)),
	]


class CamadaSoftMax(c.Structure):
	_fields_ = [
		('super', Camada),
		('kernelSoftMaxAtiva1', TOPOINTER(Kernel)),
		('kernelSoftMaxAtiva2', TOPOINTER(Kernel)),
		('kernelSoftMaxCalcGrads', TOPOINTER(Kernel)),
		('soma', TOPOINTER(Tensor)),
		('exponent', TOPOINTER(Tensor)),
	]
