import numpy as np
import copy
import re


def checkconv(layer, entrada):
	im = np.array(entrada[:-1])
	f = np.array(layer['args'][2][2])
	p = np.array(layer['args'][1][2])

	s = (im - f) / p + 1
	layer['ok'] = False
	if np.sum(np.int32(s <= 0)):
		return False
	imi = np.int64(s - 1) * p + (f - 1) - (im - 1)
	if imi[0] != 0: return False
	if imi[1] != 0: return False
	layer['saida'] = [int(s[0]), int(s[1]), layer['args'][3][2]]
	layer['ok'] = True
	return True


def checkconvNc(layer, entrada):
	im = np.array(entrada[:-1])
	p = np.array(layer['args'][1][2])
	a = np.array(layer['args'][2][2])
	f = np.array(layer['args'][3][2])
	s = ((im - 1) - (f - 1) * a) / p + 1
	layer['ok'] = False
	print(im, p, a, f, s)
	if np.sum(np.int32(s <= 0)):
		return False
	imi = np.int64(s - 1) * p + (f - 1) * a - (im - 1)
	if imi[0] != 0: return False
	if imi[1] != 0: return False
	layer['saida'] = [s[0], s[1], layer['args'][4][2]]
	layer['ok'] = True
	return True


def checkPool(layer, entrada):
	im = np.array(entrada[:-1])
	p = np.array(layer['args'][1][2])
	f = np.array(layer['args'][2][2])
	s = (im - f) / p + 1
	s = np.int32(s > 0) * s
	layer['ok'] = False
	if np.sum(np.int32(s <= 0)):
		return False
	imi = np.int64(s - 1) * p + (f - 1) - (im - 1)
	if imi[0] != 0: return False
	if imi[1] != 0: return False
	layer['saida'] = [int(s[0]), int(s[1]), entrada[-1]]
	layer['ok'] = True
	return True


def checkPadding(layer, entrada):
	layer['saida'] = [entrada[0] + layer['args'][1][2] + layer['args'][2][2],
					  entrada[1] + layer['args'][3][2] + layer['args'][3][2],
					  entrada[2]
					  ]
	layer['ok'] = True
	return True


def checkGeneric(layer, entrada):
	layer['saida'] = entrada[:]
	layer['ok'] = True
	return True


def checkFull(layer, entrada):
	layer['saida'] = [layer['args'][1][2], 1, 1]
	layer['ok'] = True
	return True


LAYERS = [
	{'name': 'Convolucao', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['passo', '2dim', [1, 1]],
		 ['filtro', '2dim', [1, 1]],
		 ['número de filtros', 'int', 1],
		 ['taxa de aprendizado', 'float', 0.1],
		 ['momento', 'float', 0.0],
		 ['decaimento de peso', 'float', 0.0]
	 ], 'generateOut': checkconv, 'ok': False
	 },
	{'name': 'ConvolucaoNcausal', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['passo', '2dim', [1, 1]],
		 ['abertura', '2dim', [1, 1]],
		 ['filtro', '2dim', [1, 1]],
		 ['número de filtros', 'int', 1],
		 ['taxa de aprendizado', 'float', 0.1],
		 ['momento', 'float', 0.0],
		 ['decaimento de peso', 'float', 0.0]
	 ], 'generateOut': checkconvNc, 'ok': False},
	{'name': 'Pooling', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['passo', '2dim', [1, 1]],
		 ['filtro', '2dim', [1, 1]],
	 ], 'generateOut': checkPool, 'ok': False},
	{'name': 'PoolingAv', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['passo', '2dim', [1, 1]],
		 ['filtro', '2dim', [1, 1]],
	 ], 'generateOut': checkPool, 'ok': False},
	{'name': 'Padding', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['top', 'int', 1],
		 ['bottom', 'int', 1],
		 ['left', 'int', 1],
		 ['right', 'int', 1],
	 ], 'generateOut': checkPadding, 'ok': False},
	{'name': 'BatchNorm', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['epsilon', 'float', 1e-10],
	 ], 'generateOut': checkGeneric, 'ok': False},
	{'name': 'SoftMax', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
	 ], 'generateOut': checkGeneric, 'ok': False},
	{'name': 'DropOut', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['probabilidade de saida', 'float', 0.5],
	 ], 'generateOut': checkGeneric, 'ok': False},
	{'name': 'FullConnect', 'saida': [0, 0, 0],
	 'args': [
		 ['typeMemory', 'options', 0, ['NO COPY', 'Shared Mem'], [0, 1]],
		 ['saida', 'int', 1],
		 ['função de ativação', 'options', 0, ['SIGMOID', 'TANH', 'RELU'], [0, 2, 4]],
		 ['taxa de aprendizado', 'float', 0.1],
		 ['momento', 'float', 0.0],
		 ['decaimento de peso', 'float', 0.0]
	 ], 'generateOut': checkFull, 'ok': False},
]

def updateDEFAULTLAYERS(layer):
	ll = None
	for i in range(len(LAYERS)):
		if layer['name'] == LAYERS[i]['name']:
			ll = LAYERS[i]
	if ll == None: return
	ll['args'] = copy.deepcopy(layer['args'])

class Arquitetura:
	VERSION = '"Gab Create Cnn: version 1.0"'
	def __init__(self):
		self.entrada = [1, 1, 1]
		self.camadas = []

	def update(self, entrada):
		self.entrada = entrada[:]
		sucess = True
		for camada in self.camadas:
			if sucess:
				sucess = camada['generateOut'](camada, entrada)
				entrada = camada['saida']
				updateDEFAULTLAYERS(camada)
			else:
				camada['ok'] = False

	def add(self, layer):
		self.camadas.append(copy.deepcopy(layer))

	def remove(self, index):
		if index < 0: index = len(self.camadas) + index
		del self.camadas[index]

	def __arg2str__(self, arg, sep=', '):
		ra = str(arg)
		if isinstance(arg, (list, tuple)):
			ra = [self.__arg2str__(value) for value in arg]
			ra = sep.join(ra)
		return ra

	def compile(self, file):
		if len(self.camadas) <= 0: return
		print('--',end='',file=file)
		print(Arquitetura.VERSION, file=file)

		print('Entrada(%d, %d, %d)' % (self.entrada[0], self.entrada[1], self.entrada[2]), file=file)
		for camada in self.camadas:
			args = [self.__arg2str__(value[2]) for value in camada['args']]
			# args.append(args.pop(0))
			print(f'{camada["name"]}(' + ', '.join(args) + ')', file=file)

	def load(self, file):
		if len(self.camadas) > 0: return -1,"Arquitetura não nula, impossivel fazer concatenação"
		lines =[line.strip() for line in open(file, 'r').readlines() if line!= ""]
		if len(lines) == 0: return -2,"Arquivo nulo"
		if not lines[0].endswith(Arquitetura.VERSION):
			return -3,f"Arquivo não compatível \n({lines[0]})\n({Arquitetura.VERSION})"
		lines.pop(0)
		for line in lines:
			fname = re.findall("^[\w]+", line)[0].strip()
			args = re.findall("\([^)]*\)", line)[0].replace("(","").replace(")","")
			args = [a.strip() for a in args.split(',') if a.strip()!=""]
			if fname == 'Entrada':
				self.entrada = [int(a) for a in args]
				continue
			# args.insert(0, args.pop())
			layer = None
			for l in LAYERS:
				if fname == l['name']:
					layer = l
					break
			if layer == None: return -4,f"Camada '{fname}' não encontrada"
			self.add(layer)
			layer = self.camadas[-1]

			i,j=0,0
			for i in range(len(layer['args'])):
				ag = layer['args'][i]
				lag = args[j]
				if ag[1]=='options' or ag[1] == 'int':
					ag[2]= int(lag)
				elif ag[1]=='float':
					ag[2]= float(lag)
				elif ag[1]=='2dim':
					ag[2]= [int(lag),int(args[j+1])]
					j+=1
				else:
					return -5, "argumento nao encontrado\n"+f"{ag[1]} {lag}"
				j+=1
		return 0,None