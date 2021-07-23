import ctypes as c
import os
import time
import numpy as np
from random import randint
from PIL import Image
import matplotlib.pyplot as plt

__dir = os.path.abspath(__file__)
__dir = os.path.realpath(__dir)
__dir = os.path.dirname(__dir)
__dll = os.path.join(__dir, '../bin/libCNNGPU.dll')
clib = c.CDLL(__dll)

queue = c.c_void_p(0)
clib.createCnnPy.argtypes = [c.c_void_p, c.c_double, c.c_double, c.c_double, c.c_uint, c.c_uint, c.c_uint]
clib.releaseCnnWrapper.argtypes = [c.c_void_p]
clib.camadaToString.argtypes = [c.c_void_p]
clib.camadaToString.restype = c.c_char_p

clib.CnnAddConvLayer.argtypes = [c.c_void_p, c.c_uint8, c.c_uint, c.c_uint, c.c_uint]
clib.CnnAddConvNcLayer.argtypes = [c.c_void_p, c.c_uint8, c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_uint,
                                   c.c_uint]
clib.CnnAddPoolLayer.argtypes = [c.c_void_p, c.c_uint8, c.c_uint, c.c_uint]
clib.CnnAddPoolAvLayer.argtypes = [c.c_void_p, c.c_uint8, c.c_uint, c.c_uint]
clib.CnnAddReluLayer.argtypes = [c.c_void_p, c.c_uint8]
clib.CnnAddPaddingLayer.argtypes = [c.c_void_p, c.c_uint8, c.c_uint, c.c_uint, c.c_uint, c.c_uint]
clib.CnnAddBatchNorm.argtypes = [c.c_void_p, c.c_uint8, c.c_double]
clib.CnnAddDropOutLayer.argtypes = [c.c_void_p, c.c_uint8, c.c_double, c.c_ulonglong]
clib.CnnAddFullConnectLayer.argtypes = [c.c_void_p, c.c_uint8, c.c_uint, c.c_uint]
clib.CnnCall.argtypes = [c.c_void_p, c.c_void_p]
clib.CnnLearn.argtypes = [c.c_void_p, c.c_void_p]
clib.CnnCalculeError.argtypes = [c.c_void_p, ]

clib.TensorGetValues.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p]
clib.TensorGetValuesOffset.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint]
clib.TensorPutValues.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p]

clib.CamadaSetParams.argtypes = [c.c_void_p, c.c_double, c.c_double, c.c_double]

clib.CnnSaveInFile.argtypes = [c.c_void_p, c.c_char_p]
clib.CnnLoadByFile.argtypes = [c.c_void_p, c.c_char_p]
clib.initRandom.argtypes = [c.c_longlong]

clib.Py_getCnnOutPutAsPPM.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p]
clib.freeP.argtypes = [c.c_void_p]

FSIGMOID = 0
FTANH = 2
FRELU = 4

TENSOR_NCPY = 0
TENSOR_HSTA = 1
TENSOR_HOST = 2

CONV = 1
POOL = 2
RELU = 3
DROPOUT = 4
FULLCONNECT = 5
SOFTMAX = 6
BATCHNORM = 7
PADDING = 8
POOLAV = 9
CONVNC = 10


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


def LCG_SEED(v):
    clib.initRandom(int(v))


class Tensor(c.Structure):
    _fields_ = [('mem', c.c_void_p),
                ('host', c.c_void_p),
                ('flag', c.c_uint8),
                ('bytes', c.c_uint),
                ('x', c.c_uint),
                ('y', c.c_uint),
                ('z', c.c_uint),
                ('w', c.c_uint),
                ]

    def __repr__(self):
        s = "Tensor[%s](%d %d %d %d)" % (["NCPY","HOST", "HSTA",  ][self.flag], self.x, self.y, self.z, self.w)
        return s

    def put(self, values: list):
        tmp = c.c_double * (self.x * self.y * self.z)
        return clib.TensorPutValues(queue, c.addressof(self), tmp(*values))

    def value(self, offset=0):
        temp = self.x * self.y * self.z
        temp = c.c_double * temp
        temp = temp(0)
        clib.TensorGetValuesOffset(queue, c.addressof(self), temp, offset * self.bytes)
        return list(temp)

    def value_np(self, offset=0):
        data = np.array(self.value(offset))
        data = data.reshape(self.z, self.x, self.y)
        return data

    def histogram(self, offset=0):
        data = self.value(offset * self.bytes)
        plt.figure()
        plt.hist(data, density=True, bins=20)
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title('%d' % (len(data),))

    def __iter__(self):
        return self.value().__iter__()


class TensorChar(Tensor):
    def put(self, values: list):
        tmp = c.c_ubyte * (self.x * self.y * self.z)
        return clib.TensorPutValues(queue, c.addressof(self), tmp(*values))

    def value(self, offset=0):
        temp = self.x * self.y * self.z
        temp = c.c_uint8 * temp
        temp = temp(0)
        clib.TensorGetValuesOffset(queue, c.addressof(self), offset * self.bytes, temp)
        return list(temp)


class Params(c.Structure):
    _fields_ = [('hitLearn', c.c_double),
                ('momento', c.c_double),
                ('decaimentoDePeso', c.c_double)
                ]


class Ponto3d(c.Structure):
    _fields_ = [('x', c.c_int),
                ('y', c.c_int),
                ('z', c.c_int)]


class GPU_ERROR(c.Structure):
    _fields_ = [('error', c.c_int),
                ('msg', c.c_char * 500)]


class Kernel(c.Structure):
    _fields_ = [('kernel', c.c_void_p),
                ('kernelName', c.c_char_p),
                ('nArgs', c.c_int),
                ('largs', c.c_void_p),
                ]


class Camada(c.Structure):
    _fields_ = [('type', c.c_uint8),
                ('flag_releaseInput', c.c_uint8),
                ('flag_notlearn', c.c_uint8),
                ('flag_usehost', c.c_uint8),
                ('parametros', Params),
                ('gradsEntrada', TOPOINTER(Tensor)),
                ('entrada', TOPOINTER(Tensor)),
                ('saida', TOPOINTER(Tensor)),
                ('queue', c.c_void_p),
                ('context', c.c_void_p),
                ('max_works', TOPOINTER(c.c_uint64)),
                ('calc_grads', c.c_void_p),
                ('corrige_pesos', c.c_void_p),
                ('ativa', c.c_void_p),
                ('release', c.c_void_p),
                ('salvar', c.c_void_p),
                ('toString', c.c_void_p),
                ('getCreateParams', c.c_void_p),
                ('setLearn', c.CFUNCTYPE(None, c.c_void_p, c.c_uint8)),
                ('setParams', c.CFUNCTYPE(None, c.c_void_p, c.c_double, c.c_double, c.c_double, )),
                ('__string__', c.c_char_p),
                ]

    def __repr__(self):
        cstr = clib.camadaToString(c.addressof(self))
        return cstr.decode('utf-8')

    def setParams(self, hitlearn, momento, decaimento):
        clib.CamadaSetParams(c.addressof(self), hitlearn, momento, decaimento)

    def setLearnable(self, canLearn: bool):
        self.setLearn(c.addressof(self), canLearn)


class CamadaConv(c.Structure):
    _fields_ = [('super', Camada),
                ('filtros', TOPOINTER(Tensor)),
                ('grad_filtros', TOPOINTER(Tensor)),
                ('grad_filtros_old', TOPOINTER(Tensor)),
                ('passo', c.c_uint),
                ('tamanhoFiltro', c.c_uint),
                ('numeroFiltros', c.c_uint),
                ('kernelConvSum', Kernel),
                ('kernelConvFixWeight', Kernel),
                ('kernelConvCalcGradsFiltro', Kernel),
                ('kernelConvCalcGrads', Kernel),
                ]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != CONV:
            raise Exception('Esperado camada Convolucional')
        cst = c.cast(LP_camada, TOPOINTER(CamadaConv))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaConvNc(c.Structure):
    _fields_ = [('super', Camada),
                ('filtros', TOPOINTER(Tensor)),
                ('grad_filtros', TOPOINTER(Tensor)),
                ('grad_filtros_old', TOPOINTER(Tensor)),
                ('passox', c.c_uint),
                ('passoy', c.c_uint),
                ('largx', c.c_uint),
                ('largy', c.c_uint),
                ('numeroFiltros', c.c_uint),
                ('kernelConvNcSum', Kernel),
                ('kernelConvNcFixWeight', Kernel),
                ('kernelConvNcCalcGradsFiltro', Kernel),
                ('kernelConvNcCalcGrads', Kernel),
                ]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != CONVNC:
            raise Exception('Esperado camada Convolucional Não causal')
        cst = c.cast(LP_camada, TOPOINTER(CamadaConvNc))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaDropOut(c.Structure):
    _fields_ = [('super', Camada),
                ('hitmap', TOPOINTER(TensorChar)),
                ('flag_releaseInput', c.c_char),
                ('p_ativacao', c.c_double),
                ('seed', c.c_ulonglong),
                ('kerneldropativa', Kernel),
                ('kerneldropcalcgrad', Kernel)]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != DROPOUT:
            raise Exception('Esperado camada DropOut')
        cst = c.cast(LP_camada, TOPOINTER(CamadaDropOut))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaFullConnect(c.Structure):
    _fields_ = [('super', Camada),
                ('z', TOPOINTER(Tensor)),
                ('pesos', TOPOINTER(Tensor)),
                ('grad', TOPOINTER(Tensor)),
                ('dz', TOPOINTER(Tensor)),
                ('fa', c.c_int),
                ('dfa', c.c_int),
                ('kernelfullfeed', Kernel),
                ('kernelfullfixWeight', Kernel),
                ('kernelfullcalcgrad1', Kernel),
                ('kernelfullcalcgrad2', Kernel)]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != FULLCONNECT:
            raise Exception('Esperado camada FullConnect')
        cst = c.cast(LP_camada, TOPOINTER(CamadaFullConnect))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaPool(c.Structure):
    _fields_ = [('super', Camada),
                ('passo', c.c_uint),
                ('tamanhoFiltro', c.c_uint),
                ('kernelPoolAtiva', Kernel),
                ('kernelPoolCalcGrads', Kernel)]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != POOL:
            raise Exception('Esperado camada FullConnect')
        cst = c.cast(LP_camada, TOPOINTER(CamadaPool))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaPoolAv(c.Structure):
    _fields_ = [('super', Camada),
                ('passo', c.c_uint),
                ('tamanhoFiltro', c.c_uint),
                ('kernelPoolAtiva', Kernel),
                ('kernelPoolCalcGrads', Kernel)]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != POOLAV:
            raise Exception('Esperado camada Pooling')
        cst = c.cast(LP_camada, TOPOINTER(CamadaPoolAv))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaRelu(c.Structure):
    _fields_ = [('super', Camada),
                ('kernelReluAtiva', Kernel),
                ('kernelReluCalcGrads', Kernel)]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != RELU:
            raise Exception('Esperado camada Relu')
        cst = c.cast(LP_camada, TOPOINTER(CamadaRelu))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaPadding(c.Structure):
    _fields_ = [('super', Camada),
                ('top', c.c_uint),
                ('bottom', c.c_uint),
                ('left', c.c_uint),
                ('right', c.c_uint),
                ]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != PADDING:
            raise Exception('Esperado camada Padding')
        cst = c.cast(LP_camada, TOPOINTER(CamadaPadding))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaSoftMax(c.Structure):
    _fields_ = [('super', Camada),
                ('kernelSoftMaxAtiva1', Kernel),
                ('kernelSoftMaxAtiva2', Kernel),
                ('kernelSoftMaxCalcGrads', Kernel),
                ('soma', TOPOINTER(Tensor)),
                ('exponent', TOPOINTER(Tensor))]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != POOL:
            raise Exception('Esperado camada SoftMax')
        cst = c.cast(LP_camada, TOPOINTER(CamadaSoftMax))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class CamadaBatchNorm(c.Structure):
    _fields_ = [('super', Camada),
                ('kernelBatchNormAtiva1', Kernel),
                ('kernelBatchNormAtiva2', Kernel),
                ('kernelBatchNormAtiva3', Kernel),
                ('kernelBatchNormAtiva4', Kernel),
                ('kernelBatchNormCalcGrads', Kernel),
                ('media', TOPOINTER(Tensor)),
                ('somaDiferenca', TOPOINTER(Tensor)),
                ('variancia', TOPOINTER(Tensor)),
                ('gradVariancia', TOPOINTER(Tensor)),
                ('epsilon', c.c_double),
                ('Y', TOPOINTER(Tensor)),
                ('B', TOPOINTER(Tensor)),
                ('gradY', TOPOINTER(Tensor)),
                ('gradB', TOPOINTER(Tensor)),
                ('diferenca', TOPOINTER(Tensor)),
                ('diferencaquad', TOPOINTER(Tensor))]

    @staticmethod
    def cast(LP_camada: TOPOINTER(Camada)):
        if not isinstance(LP_camada, TOPOINTER(Camada)):
            raise Exception('Invalid format')
        if LP_camada.type != POOL:
            raise Exception('Esperado camada BatchNorm')
        cst = c.cast(LP_camada, TOPOINTER(CamadaBatchNorm))
        return cst

    def __getattr__(self, item):
        return self.super.__getattr__(item)


class Pointer(c.Structure):
    _fields_ = [('p', c.c_void_p)]


class typeCnn(c.Structure):
    _fields_ = [('parametros', Params),
                ('camadas', c.POINTER(TOPOINTER(Camada))),
                ('lastGrad', TOPOINTER(Tensor)),
                ('target', TOPOINTER(Tensor)),
                ('size', c.c_int),
                ('sizeIn', Ponto3d),
                ('queue', c.c_void_p),
                ('cl', c.c_void_p),
                ('releaseCL', c.c_char),
                ('kernelsub', Kernel),
                ('kerneldiv', Kernel),
                ('kerneldivInt', Kernel),
                ('kernelNormalize', Kernel),
                ('kernelInt2Vector', Kernel),
                ('kernelcreateIMG', Kernel),
                ('normaErro', c.c_double),
                ('error', GPU_ERROR),
                ]


class Cnn:
    def __init__(self, x, y, z, hitlearn=0.1, momento=0.0, decaimentoDePeso=0.0):
        self.p = Pointer()
        clib.createCnnPy(c.addressof(self.p), hitlearn, momento, decaimentoDePeso, x, y, z)
        self.cnn = c.cast(self.p.p, TOPOINTER(typeCnn))
        global queue
        queue = self.queue

    def __getattr__(self, item):
        return self.cnn.__getattribute__(item)

    def __repr__(self):
        s = ''
        for i in range(self.size):
            s = s + str(self.camadas[i])
        return s

    def __del__(self):
        clib.releaseCnnWrapper(c.addressof(self.p))

    def addConv(self, passo, tamanhoFiltro, numeroFiltros, flagTensor=TENSOR_NCPY):
        return clib.CnnAddConvLayer(self.cnn, flagTensor, passo, tamanhoFiltro, numeroFiltros)

    def addConvNc(self, passox, passoy, largx, largy, filtrox, filtroy, numeroFiltros, flagTensor=TENSOR_NCPY):
        return clib.CnnAddConvNcLayer(self.cnn, flagTensor, passox, passoy, largx, largy, filtrox, filtroy,
                                      numeroFiltros)

    def addPool(self, passo, tamanhoFiltro, flagTensor=TENSOR_NCPY):
        return clib.CnnAddPoolLayer(self.cnn, flagTensor, passo, tamanhoFiltro)

    def addPoolAv(self, passo, tamanhoFiltro, flagTensor=TENSOR_NCPY):
        return clib.CnnAddPoolAvLayer(self.cnn, flagTensor, passo, tamanhoFiltro)

    def addRelu(self, flagTensor=TENSOR_NCPY):
        return clib.CnnAddReluLayer(self.cnn, flagTensor)

    def addPadding(self, top, bottom, left, right, flagTensor=TENSOR_NCPY):
        return clib.CnnAddPaddingLayer(self.cnn, flagTensor, top, bottom, left, right)

    def addBatchNorm(self, episolon=1e-12, flagTensor=TENSOR_NCPY):
        return clib.CnnAddBatchNorm(self.cnn, flagTensor, episolon)

    def addDropOut(self, ativacao, seed=time.time(), flagTensor=TENSOR_NCPY):
        return clib.CnnAddDropOutLayer(self.cnn, flagTensor, ativacao, int(seed))

    def addFullConnect(self, saida, funcaoAtivacao=FTANH, flagTensor=TENSOR_NCPY):
        return clib.CnnAddFullConnectLayer(self.cnn, flagTensor, saida, funcaoAtivacao)

    @property
    def out(self):
        return self.camadas[self.size - 1].saida.value()

    def predict(self, input_values):
        # entrada da rede
        entrada = self.camadas[0].entrada
        # tipo c de entrada
        typeIN = c.c_double * (entrada.x * entrada.y * entrada.z)

        def __predict(input_values: list):
            cinput = typeIN(*input_values)
            return clib.CnnCall(self.cnn, cinput)

        self.predict = __predict
        return __predict(input_values)

    def learn(self, target: list):
        # saida da rede
        saida = self.camadas[self.size - 1].saida
        # tipo c de saida
        typeOut = c.c_double * (saida.x * saida.y * saida.z)
        ctarget = typeOut(*target)
        return clib.CnnLearn(self.cnn, ctarget)

    def salve(self, fileName: str):
        return clib.CnnSaveInFile(self.cnn, fileName.encode('utf-8'))

    def salveOutsAsPPM(self, filename):
        p = Pointer()
        w, h = c.c_int(), c.c_int()
        clib.Py_getCnnOutPutAsPPM(self.cnn, c.addressof(p), c.addressof(h), c.addressof(w))
        img = (c.c_char_p * (h.value * w.value))(0)
        c.memmove(img, p.p, h.value * w.value)
        clib.freeP(p.p)
        if filename.endswith('.ppm'):
            with open(filename, 'wb') as f:
                f.write(b'P5 ')
                f.write(b'%d %d ' % (w.value, h.value))
                f.write(b'255 ')
                f.write(bytes(img))
        else:
            ftmp = filename + '%d.ppm' % (randint(100, 3200))
            with open(ftmp, 'wb') as f:
                f.write(b'P5 ')
                f.write(b'%d %d ' % (w.value, h.value))
                f.write(b'255 ')
                f.write(bytes(img))
            im = Image.open(ftmp)
            im.save(filename)
            del im
            os.remove(ftmp)

    @staticmethod
    def load(fileName: str):
        tmpCnn = Cnn(0, 0, 0)
        clib.CnnLoadByFile(tmpCnn.cnn, fileName.encode('utf-8'))
        return tmpCnn

    def getMSE(self):
        clib.CnnCalculeError(self.cnn)

        return self.normaErro


def testIMAGE():
    LCG_SEED(time.time())
    from random import random

    a = Cnn(2, 2, 1, hitlearn=0.1, momento=0)
    # a.addConv(1,2,3)
    a.addPadding(4, 1, 1, 1)
    # a.addFullConnect(10)
    b = [random() for _ in range(2 * 2)]
    print(a)
    a.predict(b)
    print(a.camadas[a.size - 1].entrada.value_np)
    print(a.camadas[a.size - 1].saida.value_np)
    a.salveOutsAsPPM('a.jpg')


def testXOR(iters=10):
    LCG_SEED(time.time())
    a = Cnn(2, 1, 1, hitlearn=0.1, momento=0)
    a.addFullConnect(6, FTANH)
    a.addFullConnect(5, FTANH)
    a.addFullConnect(5, FTANH)
    a.addFullConnect(5, FTANH)
    a.addFullConnect(1, FTANH)
    # a.addBatchNorm()
    entrada = [[1, 1], [1, 0], [0, 1], [0, 0]]
    saida = [[v[0] ^ v[1]] for v in entrada]
    erro = []
    j = [0, 1, 2, 3]
    import random

    for i in range(iters):
        err = 0
        # random.shuffle(j)
        for caso in j:
            a.predict(entrada[caso])
            a.learn(saida[caso])
            err += a.getMSE()
            # print(entrada[caso],saida[caso],a.out)
            # print(a.normaErro)
        erro.append(err)

    import matplotlib.pyplot as plt

    plt.plot(erro)
    plt.legend(['Erro'])
    plt.text(0, 1, str(a).replace('\t', '   '))
    plt.ylim([-0.5, 5])
    plt.show()


if __name__ == '__main__':
    testXOR(3)
