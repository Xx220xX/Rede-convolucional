import  time,pickle
from CNN_GPU.CNN_C_Wrapper import *
from pathlib import Path

FSIGMOIG = 0
FTANH = 2
FRELU = 4

REQUEST_INPUT = 0
REQUEST_GRAD_INPUT = 1
REQUEST_OUTPUT = 2
REQUEST_WEIGTH = 3

class CNN:
    def __init__(self, inputSize, hitLearn=.1, momentum=.9, weigthDecay=.5, multip=1.0):
        file = '%s/%s' % (DIR_LIBRARY, 'gpu_function.cl')
        file = file.encode('utf-8')
        self.cnn = c_Pointer()
        clib.createCnnWrapper(c.addressof(self.cnn), c.create_string_buffer(file),
                              hitLearn, momentum, weigthDecay, multip, inputSize[0], inputSize[1], inputSize[2])
        clib.initRandom(time.time_ns())

    def __del__(self):
        clib.releaseCnnWrapper(c.addressof(self.cnn))
        print('end')

    def addConvLayer(self, passo, tamanhoFitro, numeroFiltro):
        clib.CnnAddConvLayer(self.cnn.p, passo, tamanhoFitro, numeroFiltro)

    def addPoolLayer(self, passo, tamanhoFitro):
        clib.CnnAddPoolLayer(self.cnn.p, passo, tamanhoFitro)

    def addReluLayer(self):
        clib.CnnAddReluLayer(self.cnn.p)

    def addDropOutLayer(self, pontoAtivacao, seed):
        clib.CnnAddDropOutLayer(self.cnn.p, pontoAtivacao, seed)

    def addFullConnectLayer(self, saida, funcaoAtivacao):
        clib.CnnAddFullConnectLayer(self.cnn.p, saida, funcaoAtivacao)

    def predict(self, input):
        tinput = self.createInp(*input)
        clib.CnnCall(self.cnn.p, tinput)

    def learn(self, target):
        ttarg = self.targ(*target)
        clib.CnnLearn(self.cnn.p, ttarg)

    def getData(self, layer, request, nfilter=0):
        size = self.getSizeData(layer, request)
        if size is None: return None
        data = c.c_double * (size[0] * size[1] * size[2])
        data = data(0)
        err = clib.CnnGetTensorData(self.cnn.p, layer, request, nfilter, data)
        if err < 0:
            self.lastERROR = err
            return None
        return list(data)

    def getSizeData(self, layer, request):
        inx, iny, inz, n = c.c_int(0), c.c_int(0), c.c_int(0), c.c_int(0)
        err = clib.CnnGetSize(self.cnn.p, layer, request, c.addressof(inx), c.addressof(iny), c.addressof(inz),
                              c.addressof(n))
        if err < 0:
            self.lastERROR = err
            return None
        return inx.value, iny.value, inz.value, n.value

    @property
    def output(self):
        err = clib.CnnGetTensorData(self.cnn.p, -1, REQUEST_OUTPUT, 0, self.out)
        if err < 0:
            self.lastERROR = err
            return None
        return list(self.out)

    def compile(self):
        if self.error: raise Exception("ERROR")
        inx, iny, inz = c.c_int(0), c.c_int(0), c.c_int(0)
        err = clib.CnnGetSize(self.cnn.p, 0, REQUEST_INPUT, c.addressof(inx), c.addressof(iny), c.addressof(inz),
                              c.cast(0, c.c_void_p))
        if err != 0: raise Exception('Error when request input size', err)
        self.createInp = c.c_double * (inx.value * iny.value * inz.value)
        err = clib.CnnGetSize(self.cnn.p, -1, REQUEST_OUTPUT, c.addressof(inx), c.addressof(iny), c.addressof(inz),
                              c.cast(0, c.c_void_p))
        if err != 0: raise Exception('Error when request output size', err)
        self.out = c.c_double * (inx.value * iny.value * inz.value)
        self.targ = self.out
        self.out = self.out(0)

    def info(self):
        clib.CnnInfo(self.cnn.p)

    def save(self, fileName:str):
        filedesc = Path(fileName).with_suffix('.cdc')
        self.salveCnnDescriptor(filedesc)
        fileName = fileName.encode('utf-8')
        return clib.CnnSaveInFile(self.cnn.p, c.create_string_buffer(fileName))

    @staticmethod
    def load(fileName):
        self = CNN([2,2,1])
        fileName = fileName.encode('utf-8')
        clib.CnnLoadByFile(self.cnn.p, c.create_string_buffer(fileName))
        self.compile()
        return self
    @property
    def error(self):
        return clib.getCnnError(self.cnn.p)
    @property
    def errorMsg(self):
        buff = c.create_string_buffer(''.encode('utf-8'),255)
        clib.getCnnErrormsg(self.cnn.p,buff)
        return  buff.value.decode('utf-8')
    def salveCnnDescriptor(self,file):
       desc_c = c_Pointer()
       clib.generateDescriptor(c.addressof(desc_c),self.cnn.p)
       msg = c.cast(desc_c.p,c.c_char_p)
       msg = msg.value.decode('utf-8')
       clib.freeP(desc_c.p)
       desc = eval(msg)
       with open(file,'wb') as f:
           pickle.dump(desc,f)
    # AUXILIAR FUNCTION
    def getOutputAsIndexMax(self):
        ans = clib.CnnGetIndexMax(self.cnn.p)
        return ans
    def normalizeVector(self,vector:list,maxOutput,minOutput):
        out_TYPE =c.c_double * len(vector)
        inp = out_TYPE(*vector)
        out = out_TYPE()
        clib.normalizeGPU(self.cnn.p,inp,out,len(vector),maxOutput,minOutput)
        return list(out)
    def normalizeVectorKnowedSpace(self,vector:list,maxInput,minInput,maxOutput,minOutput):
        out_TYPE =c.c_double * len(vector)
        tmp_inp = out_TYPE(*vector)
        tmp_out = out_TYPE(*vector)
        clib.normalizeGPUSpaceKnow(self.cnn.p,tmp_inp,tmp_out,len(vector),maxInput,minInput,maxOutput,minOutput)
        return list(tmp_out)
    def getOutPutAsPPM(self):
        p = c_Pointer()
        h = c.c_size_t()
        w = c.c_size_t()
        clib.Py_getCnnOutPutAsPPM(self.cnn.p, c.addressof(p), c.addressof(h), c.addressof(w))
        h = h.value
        w = w.value
        out = (c.c_ubyte*(w*h))()
        c.memmove(out,p.p,w*h)
        a = bytes(out)
        clib.freeP(p.p)
        return (h,w,a)