import ctypes as c
import os
from platform import architecture
clib = None
__dir = ''
DIR_LIBRARY = ''
if architecture()[0] == '64bit':
    temp = os.path.abspath(__file__)
    temp = os.path.realpath(temp)
    temp = os.path.dirname(temp)
    # temp = os.path.join(temp, "lib/gabriela6.dll")
    __dir = temp
    DIR_LIBRARY = os.path.join(temp, "lib")
    temp = os.path.join(temp, "lib/libCNNGPU.dll")
    clib = c.CDLL(temp)
else:
    raise Exception('unsuport 32 bits architecture ')

#Cnn createCnnPythonWrapper(char *kernelFile, double hitLearn, double momento, double decaimentoDePeso, double multiplicador, UINT inx, UINT iny, UINT inz) {
clib.createCnnWrapper.argtypes = [c.c_void_p,c.c_void_p,
                                        c.c_double, c.c_double, c.c_double, c.c_double,
                                        c.c_int,c.c_int,c.c_int]

#void releaseCnn(Cnn *pc) {
clib.releaseCnnWrapper.argtypes = [c.c_void_p]

#int CnnAddConvLayer(Cnn c, UINT passo, UINT tamanhoDoFiltro, UINT numeroDeFiltros) {
clib.CnnAddConvLayer.argtypes = [c.c_void_p,c.c_uint,c.c_uint,c.c_uint]
clib.CnnAddPoolLayer.argtypes = [c.c_void_p,c.c_uint,c.c_uint]
clib.CnnAddReluLayer.argtypes = [c.c_void_p]
clib.CnnAddDropOutLayer.argtypes = [c.c_void_p,c.c_double,c.c_int64]
clib.CnnAddFullConnectLayer.argtypes = [c.c_void_p,c.c_uint,c.c_int]

clib.CnnCall.argtypes = [c.c_void_p,c.c_void_p]
clib.CnnLearn.argtypes = [c.c_void_p,c.c_void_p]
clib.CnnGetSize.argtypes = [c.c_void_p,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_void_p,c.c_void_p]
clib.CnnGetTensorData.argtypes = [c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_void_p]
clib.initRandom.argtypes = [c.c_int64]

clib.CnnSaveInFile.argtypes = [c.c_void_p,c.c_void_p]
clib.CnnLoadByFile.argtypes = [c.c_void_p,c.c_void_p]
clib.openFILE.argtypes = [c.c_void_p,c.c_void_p,c.c_void_p]
clib.closeFile.argtypes = [c.c_void_p]

clib.freeP.argtypes = [c.c_void_p]
clib.generateDescriptor.argtypes = [c.c_void_p,c.c_void_p]

clib.CnnInfo.argtypes = [c.c_void_p]
clib.getCnnError.argtypes = [c.c_void_p]
clib.getCnnErrormsg.argtypes = [c.c_void_p,c.c_void_p]

# auxiliares
clib.CnnGetIndexMax.argtypes = [c.c_void_p]
clib.normalizeGPU.argtypes = [c.c_void_p,c.c_void_p,c.c_void_p,c.c_int,c.c_double,c.c_double]

class c_Pointer(c.Structure):
    _fields_ = [("p", c.c_void_p)]
