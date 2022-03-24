import ctypes
import random
import time
from threading import Thread

dll_path = r"C:\Users\hslhe\CLionProjects\Rede-convolucional\Gab\bin\libgab_library_cnn.dll"
gab_dll = ctypes.CDLL(dll_path)

########## Funções ############

'''
int PY_Cnn_new(Cnn *self);

int PY_Cnn_release(Cnn self);

void PY_Cnn_out(Cnn self, float *p);

int PY_Cnn_lua(Cnn self, char *luaCommand);

int PY_Cnn_train(Cnn self, int epoca, int nbatch, float *input_values, float *target_values, int nsamples, Info_callback *info);

int PY_Cnn_force_end(Cnn self);

int PY_Cnn_predict(Cnn self, float *input_value, float *answer);

int PY_Cnn_seed(unsigned int long long  seed);
'''


class Info(ctypes.Structure):
	_fields_ = [
		('epoca', ctypes.c_int),
		('image', ctypes.c_int),
		('progress', ctypes.c_float),
		('ecx', ctypes.c_float),
		('a', ctypes.c_float),
		('b', ctypes.c_float),
	]
	epoca: ctypes.c_int
	image: ctypes.c_int
	progress: ctypes.c_float
	ecx: ctypes.c_float
	a: ctypes.c_float
	b: ctypes.c_float


cCnn_p = ctypes.c_void_p
cfloat_p = ctypes.c_void_p
cchar_p = ctypes.c_void_p
cint = ctypes.c_int32
cInfo_p = ctypes.c_void_p
cullint = ctypes.c_uint64

gab_dll.PY_Cnn_new.argtypes = [cCnn_p]
gab_dll.PY_Cnn_release.argtypes = [cCnn_p]
gab_dll.PY_Cnn_out.argtypes = [cCnn_p, cfloat_p]
gab_dll.PY_Cnn_lua.argtypes = [cCnn_p, cchar_p]
gab_dll.PY_Cnn_train.argtypes = [cCnn_p, cint, cint, cfloat_p, cfloat_p, cint, cInfo_p]
gab_dll.PY_Cnn_force_end.argtypes = [cCnn_p]
gab_dll.PY_Cnn_predict.argtypes = [cCnn_p, cfloat_p, cfloat_p]
gab_dll.PY_Cnn_seed.argtypes = [cullint]
gab_dll.PY_Cnn_print.argtypes = [cCnn_p]


def list2floatp(lista):
	l = len(lista)
	t = ctypes.c_float * l
	return t(*lista)


class Cnn:
	def __init__(self, s_in, sout):
		self.core = ctypes.c_void_p(0)
		gab_dll.PY_Cnn_new(ctypes.addressof(self.core))
		self.s_in = s_in
		self.s_out = sout
		self.Entrada(*s_in)

	def __del__(self):
		gab_dll.PY_Cnn_release(self.core)

	def lua(self, command: str):
		command = command.encode('utf-8')
		command = ctypes.create_string_buffer(command)
		gab_dll.PY_Cnn_lua(self.core, command)

	def show(self):
		gab_dll.PY_Cnn_print(self.core)

	def Entrada(self, x=1, y=1, z=1):
		self.lua(f'Entrada({x}, {y}, {z})')
		self.s_in = (x, y, z)

	def treinar(self, entradas_saidas, epocas, batch, fshow=None):
		i = Info(0, 0, 0, 1, 0.99999)
		nsamples = len(entradas_saidas)
		input_values, target_values = [], []
		for inp, out in entradas_saidas:
			input_values.extend(inp)
			target_values.extend(out)
		input_values = list2floatp(input_values)
		target_values = list2floatp(target_values)
		runing = True

		def show(progresso, ep_atual, im_atual, ep_total, im_total, ecx, dt):
			print(f'\r Treinando {"%.2f" % (progresso,)}%  {ep_atual}/{ep_total}', end='')
			t = 'calculando' if progresso == 0 else '-%.1f seg' % (dt * (100 / progresso - 1),)
			print(f' Erro %.4f  %s  ' % (ecx, t), end='')

		fshow = fshow if fshow else show
		t0 = time.time()

		def up():
			while runing:
				fshow(float(i.progress), i.epoca, i.image, epocas, nsamples, i.ecx, time.time() - t0)
				time.sleep(0.1)
			fshow(float(i.progress), i.epoca, i.image, epocas, nsamples, i.ecx, time.time() - t0)

		th = Thread(target=up, daemon=True)
		th.start()
		try:
			gab_dll.PY_Cnn_train(self.core, epocas, batch, input_values, target_values, nsamples, ctypes.addressof(i))
		# gab_dll.PY_Cnn_train(self.core, epocas, batch, input_values, target_values, nsamples, 0)
		except Exception as e:
			runing = False
			time.sleep(1e-2)
			print()
			print(e)
		runing = False
		th.join(1)

	def predict(self, entrada) -> list:
		entrada = list2floatp(entrada)
		saida = list2floatp([0] * self.s_out[0])
		gab_dll.PY_Cnn_predict(self.core, entrada, saida)
		return list(saida)


def arch(cnn: Cnn, saida):
	arquitetura = f'''
	lr = 1e-3
	mom = 0
	wdec = 0
	epsilon = 1e-7
	ConvolucaoF(P2D(1), P3D(3, 3, 8), FRELU, P2D(1), P2D(1), Params(lr, mom, wdec, 0))
	ConvolucaoF(P2D(1), P3D(3, 3, 12), FRELU, P2D(1), P2D(1), Params(lr, mom, wdec, 0))
	ConvolucaoF(P2D(1), P3D(3, 3, 16), FRELU, P2D(0), P2D(0), Params(lr, mom, wdec, 0))
	FullConnect(20, FRELU, Params(lr, mom, wdec, 0))
	FullConnect({saida}, FSOFTMAX(epsilon), Params(lr, mom, wdec, 0))
	'''
	cnn.lua(arquitetura)


random.seed(10)
a = Cnn((5, 5, 1), (2,))
arch(a, 2)
# a.show()
# print('-' * 40)
samples = 1000
testes = 100
sz = 25
so = 2
data = []
for i in range(samples):
	w = [random.random() for x in range(sz)]
	data.append((w, [1, 0] if sum(w) / len(w) > 0.4 else [0, 1]))
a.treinar(data, epocas=100, batch=10)
data = []
for i in range(testes):
	w = [random.random() for x in range(sz)]
	data.append((w, [1, 0] if sum(w) / len(w) > 0.5 else [0, 1]))
ecx = 0
print()
for inp, targ in data:
	saida = a.predict(inp)
	ecx += ((targ[0] - saida[0]) ** 2 + (targ[1] - saida[1])) / 2
	# saida = [1 if saida[0] > saida[1] else 0]
	# saida.append(1 - saida[0])
	# print(targ, saida )

print(ecx / testes)
