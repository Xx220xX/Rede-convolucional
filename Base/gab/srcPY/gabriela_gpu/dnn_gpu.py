# https://medium.com/binaryandmore/beginners-guide-to-deriving-and-implementing-backpropagation-e3c1a5a1e536
from gabriela_gpu.dnnCWrapper import *
ALAN     = 0
TANH     = 1
RELU     = 2
SIGMOID  = 3
SOFTMAX  = 4
IDENTIFY = 5

class DNN:  # deep neural network
    def __init__(self, arch, hitLearn=0.01,funcoesDeAtivacao=None,normalizar=None):
        self.n = arch
        self.gab = c_Gab()
        self.vetInp, self.vetOut = c.c_double * arch[0], c.c_double * arch[-1]
        n = c.c_int * len(arch)
        n, ln = n(*arch), len(arch)
        self.sizeOut = arch[-1]
        self.gab_p = c.addressof(self.gab)
        if funcoesDeAtivacao == None:
            funcoesDeAtivacao = [TANH]*(ln-1)
        if normalizar == None:
            normalizar = [0]*(ln-1)
        funcs = c.c_int*(ln-1)
        norm = funcs(*normalizar)
        funcs = funcs(*funcoesDeAtivacao)
        clib.create_DNN(self.gab_p, n, ln,funcs,norm, hitLearn)
        self.ot = self.vetOut()
    def __call__(self, input):
        temp = self.vetInp(*input)
        clib.call(self.gab_p, temp)
    @property
    def out(self):
        clib.getoutput(self.gab_p, self.ot)
        return [self.ot[i] for i in range(self.sizeOut)]

    def aprender(self, true_output):
        temp = self.vetOut(*true_output)
        clib.learn(self.gab_p, temp)

    def learn(self, true_output):
        temp = self.vetOut(*true_output)
        clib.learn(self.gab_p, temp)

    def __del__(self):
        print('release gab')
        clib.release(self.gab_p)

    def save(self, file2save):
        raise ModuleNotFoundError("function not find")

    @staticmethod
    def load(file2load):
        raise ModuleNotFoundError("function not find")

    def setHitlearn(self, hl):
        clib.sethitlearn(self.gab_p, hl)

    def randomize(self):
        clib.randomize(self.gab_p)
    def getA(self,l):
        vet = c.c_double*self.n[l]
        vet = vet()
        clib.getA(self.gab_p,l,vet)
        return list(vet)