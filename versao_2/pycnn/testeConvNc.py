from CNN import *
rede = Cnn(4,4,2,0.1,0)
LCG_SEED(time.time())
rede.addConvNc(1,1,2,2,2,2,2)
entrada = [i/33 for i in range(1,32+1)]
saida = [(9 - i)/9 for i in range(1,9)]
er = {'da':[],'ds':[],'dw':[],'w':[]}
cnc = CamadaConvNc.cast(rede.camadas[0])
for i in range(10000):
    rede.predict(entrada)
    rede.learn(saida)
    er['dw'].append(np.linalg.norm(cnc.grad_filtros.value() + cnc.grad_filtros.value(1)))
    er['w'].append(np.linalg.norm(cnc.filtros.value() + cnc.filtros.value(1)))
    er['da'].append(np.linalg.norm(rede.camadas[0].gradsEntrada.value()))
    er['ds'].append(rede.normaErro)


import matplotlib.pyplot as plt
for k,v in er.items():
    plt.plot(v,label=k)

plt.legend()
plt.xlabel('epocas')
plt.ylabel('Potencia media')
plt.title('Teste de convergencia camada conv non-causal')
rede.salveOutsAsPPM('treinado.png')
print(rede.camadas[0].saida.value())
print(saida)
plt.show()