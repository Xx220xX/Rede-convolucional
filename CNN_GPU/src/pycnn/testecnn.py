from CNN import *

entrada = [1,1]
pesos = [0.5 , 0.2, 0.7,0.8]
objetivo = [0.52,  0.8]
h = 0.5
LCG_SEED(1)
cnn = Cnn(2, 1, 1, hitlearn=h, momento=0)
cnn.addFullConnect(2)
cfc = CamadaFullConnect.cast(cnn.camadas[0])
cfc.pesos.put(pesos)

a = np.array(entrada).reshape(2, 1)
w = np.array(pesos).reshape(2, 2)
t = np.array(objetivo).reshape(2, 1)


def iter():
    global a
    global w
    cnn.predict(entrada)
    z = w.dot(a)
    s = 1 / (1 + np.exp(-z))

    e = s - t
    dz = e * (s * (1 - s))
    dw = dz.dot(a.transpose())
    da = w.transpose().dot(dz)

    #
    # print('w')
    # print('cnn', cfc.pesos.value_np)
    # print('py', w)
    # print()
    #
    # print('z')
    # print('cnn', cfc.z.value_np)
    # print('py', z)
    # print('dif', cfc.z.value_np-z)
    # print()
    #
    # print('s')
    # print('cnn', cfc.super.saida.value_np)
    # print('py', s)
    # print('dif', cfc.super.saida.value_np - s)
    # print('objetivo', t)
    # print()
    cnn.learn(objetivo)

    print('e')
    print('cnn', cnn.lastGrad.value_np)
    print('py', e)
    print('dif', cnn.lastGrad.value_np - e)
    #
    # print()
    #
    # print('dz')
    # print('cnn', cfc.dz.value_np)
    # print('py', dz)
    # print('dif', cfc.dz.value_np - dz)
    # print()
    #
    #
    # print('da')
    # print('cnn', cfc.super.gradsEntrada.value_np)
    # print('py', da)
    # print('dif', cfc.super.gradsEntrada.value_np - da)
    # print()
    w = w - h * dw
    # print('w2')
    # print('cnn', cfc.pesos.value_np)
    # print('py', w)
    # print('dif', cfc.pesos.value_np - w)
    # print('-'*50)
    # print()




for i in range(10000):
    iter()

