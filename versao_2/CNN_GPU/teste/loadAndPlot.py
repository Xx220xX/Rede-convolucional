import numpy as np

# dir = "D:/Henrique/treino_ia/treino_numero_0_9/statistic/"
dir = './statistic'
import os, ctypes as c
import matplotlib.pyplot as plt
import numpy as np
files = [dir + x for x in os.listdir(dir)]
geralx = []
geraly = []
geralt = []
i = 0
for file in files:
	if not file.endswith('.bin'): continue
	i+=1
	#if i == 30:break
	with open(file, 'rb') as f:
		b = f.read(8)
		size_t = c.c_char * 8
		b = size_t(*b)
		length = c.cast(b, c.POINTER(c.c_size_t))[0]
		x = f.read(length * 8)
		y = f.read(length * 8)
		v_c = c.c_char * (length * 8)
		x = c.cast(v_c(*x), c.POINTER(c.c_double))
		y = c.cast(v_c(*y), c.POINTER(c.c_double))
		x = [x[i] for i in range(1000,length)]
		y = [y[i] for i in range(1000,length)]
		#plt.figure(file)

		#plt.title(file.replace(dir, 'epoca ').replace('.bin', ''))
		#plt.plot(x)
		#plt.plot(y)
		#plt.legend(['erro medio', 'acerto medio'])

		geralx.extend(x)# sum(x) / len(x))
		geraly.extend(y)#sum(y) / len(y))
		t0 = 0
		if len(geralt)>0:t0 = geralt[-1]
		geralt.extend(list(t0+np.arange(0,1,1/len(y))))#(len(geralt) + 1)
plt.figure('Final')
plt.title('Aprendizado')
plt.plot(geralt, geralx)
plt.plot(geralt, geraly)
plt.xlabel('epoca')
plt.legend(['Erro', 'Taxa Acerto'])

plt.show()

