from CNN import *
rede = Cnn(5,1,1)
rede.addDropOut(0.5,time.time())
print(rede)
entrada = [i/6 for i in range(1,5+1)]
rede.predict(entrada)
cdo = CamadaDropOut.cast(rede.camadas[0])
print(cdo.seed)
print(cdo.super.entrada.value())
print(cdo.hitmap.value())
print(cdo.super.saida.value())
