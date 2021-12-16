import random
import socket
import time
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from threading import Thread

root = tk.Tk()
dimensao = (720, 540)
root.geometry("%dx%d" % dimensao)
#  colocar graficos
figure = plt.Figure(figsize=(6, 6), dpi=100)
ax = [figure.add_subplot(3, 4, x) for x in range(1, 11)]
cv = FigureCanvasTkAgg(figure, root)
cv.get_tk_widget().pack(fill=tk.BOTH)
i = 1
for axe in ax:
	axe.set_title('%d' % (i,))
	axe.grid()
	i += 1


def bar(id, x):
	# print(id, len(ax))
	axe = ax[id - 1]
	weights = np.ones_like(x) / len(x)
	# print(id ,"u = %f, o = %f"%(np.mean(x),np.std(x)))
	axe.clear()
	axe.hist(x,bins=50, weights=weights)
	axe.set_ylim([0, 0.5])


def recvcdata(conn, n, tipo, bytes):
	return np.frombuffer(conn.recv(int(n * bytes)), dtype=tipo)


ctipos = [np.double, np.float32, np.int32, np.int8]
csizeof = [8, 4, 4, 4]


def receber(conn: socket.socket):
	id = recvcdata(conn, 1, np.int8, 1)[0]
	length = recvcdata(conn, 1, np.uint64, 8)[0]
	x = recvcdata(conn, 1, np.uint64, 8)[0]
	y = recvcdata(conn, 1, np.uint64, 8)[0]
	z = recvcdata(conn, 1, np.uint64, 8)[0]
	w = recvcdata(conn, 1, np.uint64, 8)[0]
	tipo = recvcdata(conn, 1, np.uint8, 1)[0]
	dados = recvcdata(conn, length, ctipos[tipo], csizeof[tipo])
	return id, dados


import socket


def tcp():
	HOST = '127.0.0.1'
	PORT = 8080
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind((HOST, PORT))
		s.listen()
		while True:
			conn, addr = s.accept()
			with conn:
				print('Connected by', addr)
				try:
					i = 0
					while True:
						id, data = receber(conn)
						bar(id - 1, data)
						i += 1
						if i >= 10:
							cv.draw()
							i = 0
				except Exception as e:
					print("desconectado:\n",e)


# time.sleep(0.001)

# tcp()
Thread(target=tcp, daemon=True).start()
root.mainloop()
