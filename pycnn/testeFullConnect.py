from CNN import *

LCG_SEED(10)
np.random.seed(10)


class FCL:
    def __init__(self, entrada, saida, hitlearn,w=None):
        self.M = entrada
        self.N = saida
        if w is None:
            self.w = np.random.random(self.N * self.M).reshape(self.N, self.M)
        else:
            self.w = w.reshape(self.N,self.M)
        self.a = None
        self.z = None
        self.s = None
        self.dz = None
        self.dw = None
        self.ds = None
        self.da = None
        self.f = np.tanh
        self.h = hitlearn

        def dftanh(x):
            y = np.tanh(x)
            return (1 - y * y)

        self.df = dftanh

    def feed(self, entrada):
        self.a = entrada.reshape(self.M, 1)
        self.z = self.w.dot(self.a)
        self.s = self.f(self.z)

    def back(self, target, ds=None):
        if ds is None:
            ds = self.s - target.reshape(self.N, 1)
        self.ds = ds
        self.dz = self.df(self.z) * self.ds
        self.da = self.w.transpose().dot(self.dz)
        self.dw = self.dz.dot(self.a.transpose())
        self.w = self.w - self.h * self.dw


class PYFNN:
    def __init__(self, M, hitLean=0.1):
        self.M = M
        self._M = M
        self.h = hitLean
        self.camadas = []

    def add(self, N,w=None):
        self.camadas.append(FCL(self._M, N, self.h,w))
        self._M = N
        return self.camadas[-1]

    def predict(self, entrada):
        entrada = np.array(entrada)
        for cm in self.camadas:
            cm.feed(entrada)
            entrada = cm.s

    def learn(self, target):
        target =np.array(target)
        ds = None
        for cm in self.camadas[::-1]:
            cm.back(target, ds)
            ds = cm.da


    def getMSE(self):
        m = self.camadas[-1].ds
        return np.linalg.norm(m.reshape(m.size))
    def __repr__(self):
        s = ''
        for cm in self.camadas:
            s += 'FULLCONNECT %d %d '%(cm.M,cm.N) + f'{cm.w.shape}' +'\n'
        return s

entrada = [[1, 1], [1, 0], [0, 1], [0, 0]]
saida = [[v[0] ^ v[1]] for v in entrada]

cnn = Cnn(2, 1, 1, 0.1, 0, 0)
pyn = PYFNN(2, 0.1)

cnn.addFullConnect(5, FTANH)
pyn.add(5, CamadaFullConnect.cast(cnn.camadas[0]).pesos.value_np())

cnn.addFullConnect(5, FTANH)
pyn.add(5, CamadaFullConnect.cast(cnn.camadas[1]).pesos.value_np())

cnn.addFullConnect(1, FTANH)
pyn.add(1,CamadaFullConnect.cast(cnn.camadas[2]).pesos.value_np())

print(pyn)
erro = [[], []]
for epoca in range(100):
    epy, egab = 0, 0
    for i in range(len(entrada)):
        cnn.predict(entrada[i])
        pyn.predict(entrada[i])
        cnn.learn(saida[i])
        pyn.learn(saida[i])
        egab += cnn.getMSE()
        epy += pyn.getMSE()
        # print('-'*10)
        # print('pyn',pyn.camadas[2].da)
        # print('cnn',CamadaFullConnect.cast(cnn.camadas[2]).super.gradsEntrada.value_np())

    erro[0].append(egab / len(entrada))
    erro[1].append(epy / len(entrada))
if len(erro[0])<2:
    exit(0)
plt.plot(erro[0])
plt.plot(erro[1])
plt.xlabel('epoca')
plt.ylabel('Erro')
plt.legend(['Gabriela', 'numpy'])
plt.show()
