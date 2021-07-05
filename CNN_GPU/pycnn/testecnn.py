from CNN import *

rede = Cnn.load('numeros0_9.cnn')

#conv = rede.camadas[0]
#conv = CamadaConv.cast(conv)

entrada = [1]*(28*28*1)
rede.predict(entrada)
rede.salveOutsAsPPM('redeteste.jpg')

for i in range(rede.size):
    cm = rede.camadas[i]
    cm.saida.histogram()
plt.show()


