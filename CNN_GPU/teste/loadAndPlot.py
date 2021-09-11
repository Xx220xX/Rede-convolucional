import numpy as np

dir = "D:/Henrique/treino_ia/treino_numero_0_9/statistic/"

import os, ctypes as c
import matplotlib.pyplot as plt

files = [dir + x for x in os.listdir(dir)]
geralx = []
geraly = []
geralt = []
for file in files:
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
		x = [x[i] for i in range(length)]
		y = [y[i] for i in range(length)]
		# plt.figure(file)
		# plt.title(file.replace(dir, 'epoca ').replace('.bin', ''))
		# plt.plot(x)
		# plt.plot(y)
		# plt.legend(['erro medio', 'acerto medio'])
		# global geralx
		# global geralt
		# global geraly
		geralx = geralx + x
		geraly = geraly + y
		t = np.arange(0, 1, 1 / length)
		if len(geralt) != 0:
			t = t + geralt[-1]
		geralt = geralt + list(t)
plt.title('Aprendizado')
plt.plot(geralt,geralx)
plt.plot(geralt,geraly)
plt.legend(['Erro','Taxa Acerto'])
plt.show()
