import ctypes as c

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
		('kernel',c.c_void_p),
		('kernel_name',c.c_char_p),
		('nArgs',c.c_int),
		('l_args',c.POINTER(c.c_size_t)),
	]
class Exception(c.Structure):
	_fields_ = [
		('error',c.c_int),
		('msg',c.c_char),
		('',EXCEPTION_MAX_MSG_SIZE),
	]
class Ponto3d(c.Structure):
	_fields_ = [
		('x',c.c_void_p),
		('y',c.c_void_p),
		('z',c.c_void_p),
	]
class Tensor(c.Structure):
	_fields_ = [
		('data',c.c_void_p),
		('host',c.c_void_p),
		('flag',c.c_char),
		('bytes',c.c_void_p),
		('x',c.c_void_p),
		('y',c.c_void_p),
		('z',c.c_void_p),
		('w',c.c_void_p),
	]
class TensorC(c.Structure):
	_fields_ = [
		('x',c.c_int),
		('y',c.c_int),
		('z',c.c_int),
		('l',c.c_int),
		('data',c.POINTER(c.c_double)),
	]
class Params(c.Structure):
	_fields_ = [
		('hitLearn',c.c_double),
		('momento',c.c_double),
		('decaimentoDePeso',c.c_double),
	]
class Camada(c.Structure):
	_fields_ = [
		('type',c.c_char),
		('flag_releaseInput',c.c_char),
		('flag_notlearn',c.c_char),
		('flag_usehost',c.c_char),
		('parametros',Params),
		('gradsEntrada',TOPOINTER(Tensor)),
		('entrada',TOPOINTER(Tensor)),
		('saida',TOPOINTER(Tensor)),
		('queue',c.c_void_p),
		('context',c.c_void_p),
		('max_works',c.POINTER(c.c_size_t)),
		('calc_grads',c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_void_p)),
		('corrige_pesos',c.CFUNCTYPE(c.c_int,c.c_void_p)),
		('ativa',c.CFUNCTYPE(c.c_int,c.c_void_p)),
		('release',c.CFUNCTYPE(c.c_int,c.c_void_p)),
		('salvar',c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_void_p,c.c_void_p,c.c_void_p)),
		('toString',c.CFUNCTYPE(c.c_char_p,c.c_void_p)),
		('getCreateParams',c.CFUNCTYPE(c.c_char_p,c.c_void_p)),
		('setLearn',c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_char)),
		('setParams',c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_double,c.c_double,c.c_double)),
		('__string__',c.c_char_p),
	]
class CamadaBatchNorm(c.Structure):
	_fields_ = [
		('super',Camada),
		('kernelBatchNormAtiva1',Kernel),
		('kernelBatchNormAtiva2',Kernel),
		('kernelBatchNormAtiva3',Kernel),
		('kernelBatchNormAtiva4',Kernel),
		('kernelBatchNormCalcGrads1',Kernel),
		('kernelBatchNormCalcGrads2',Kernel),
		('kernelBatchNormCorrige',Kernel),
		('epsilon',c.c_double),
		('media',TOPOINTER(Tensor)),
		('somaDiferenca',TOPOINTER(Tensor)),
		('variancia',TOPOINTER(Tensor)),
		('gradVariancia',TOPOINTER(Tensor)),
		('Y',TOPOINTER(Tensor)),
		('B',TOPOINTER(Tensor)),
		('gradY',TOPOINTER(Tensor)),
		('gradB',TOPOINTER(Tensor)),
		('diferenca',TOPOINTER(Tensor)),
		('diferencaquad',TOPOINTER(Tensor)),
		('norma',TOPOINTER(Tensor)),
	]
class CamadaConv(c.Structure):
	_fields_ = [
		('super',Camada),
		('filtros',TOPOINTER(Tensor)),
		('grad_filtros',TOPOINTER(Tensor)),
		('grad_filtros_old',TOPOINTER(Tensor)),
		('passo',c.c_void_p),
		('tamanhoFiltro',c.c_void_p),
		('numeroFiltros',c.c_void_p),
		('kernelConvSum',Kernel),
		('kernelConvFixWeight',Kernel),
		('kernelConvCalcGradsFiltro',Kernel),
		('kernelConvCalcGrads',Kernel),
	]
class CamadaConvNc(c.Structure):
	_fields_ = [
		('super',Camada),
		('filtros',TOPOINTER(Tensor)),
		('grad_filtros',TOPOINTER(Tensor)),
		('grad_filtros_old',TOPOINTER(Tensor)),
		('passox',c.c_void_p),
		('passoy',c.c_void_p),
		('largx',c.c_void_p),
		('largy',c.c_void_p),
		('numeroFiltros',c.c_void_p),
		('kernelConvNcSum',Kernel),
		('kernelConvNcFixWeight',Kernel),
		('kernelConvNcCalcGradsFiltro',Kernel),
		('kernelConvNcCalcGrads',Kernel),
	]
class CamadaDropOut(c.Structure):
	_fields_ = [
		('super',Camada),
		('hitmap',TOPOINTER(Tensor)),
		('flag_releaseInput',c.c_char),
		('p_ativacao',c.c_double),
		('seed',c.c_uint64),
		('kerneldropativa',Kernel),
		('kerneldropcalcgrad',Kernel),
	]
class CamadaFullConnect(c.Structure):
	_fields_ = [
		('super',Camada),
		('z',TOPOINTER(Tensor)),
		('pesos',TOPOINTER(Tensor)),
		('grad',TOPOINTER(Tensor)),
		('dz',TOPOINTER(Tensor)),
		('fa',c.c_int),
		('dfa',c.c_int),
		('kernelfullfeed',Kernel),
		('kernelfullfixWeight',Kernel),
		('kernelfullcalcgrad1',Kernel),
		('kernelfullcalcgrad2',Kernel),
	]
class CamadaPadding(c.Structure):
	_fields_ = [
		('super',Camada),
		('top',c.c_size_t),
		('bottom',c.c_size_t),
		('left',c.c_size_t),
		('right',c.c_size_t),
		('ativa',Kernel),
		('calcGrad',Kernel),
	]
class CamadaPool(c.Structure):
	_fields_ = [
		('super',Camada),
		('passo',c.c_void_p),
		('tamanhoFiltro',c.c_void_p),
		('kernelPoolAtiva',Kernel),
		('kernelPoolCalcGrads',Kernel),
	]
class CamadaPoolAv(c.Structure):
	_fields_ = [
		('super',Camada),
		('passo',c.c_void_p),
		('tamanhoFiltro',c.c_void_p),
		('kernelPoolAvAtiva',Kernel),
		('kernelPoolAvCalcGrads',Kernel),
	]
class CamadaRelu(c.Structure):
	_fields_ = [
		('super',Camada),
		('kernelReluAtiva',Kernel),
		('kernelReluCalcGrads',Kernel),
	]
class CamadaSoftMax(c.Structure):
	_fields_ = [
		('super',Camada),
		('kernelSoftMaxAtiva1',Kernel),
		('kernelSoftMaxAtiva2',Kernel),
		('kernelSoftMaxCalcGrads',Kernel),
		('soma',TOPOINTER(Tensor)),
		('exponent',TOPOINTER(Tensor)),
	]
