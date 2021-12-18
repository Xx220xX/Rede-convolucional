import ctypes
import re
import sys


def loadFile(file):
	return open(file, 'r').read()


def removeComments(header):
	s = re.sub("\\/\\/[^\n]*", "", header)
	s = re.sub("\/\*(.|\n)*(?<=\*\/)", "", s)
	return s


def removeSpaces(header):
	header = header.split('\n')
	headers = []
	for i in range(len(header)):
		s = header[i]
		s = re.sub("\\t", " ", s)
		s = re.sub("[\\s]{2,}", " ", s)
		s = re.sub("^\\s", "", s)
		if s != '':
			headers.append(s)
	s = '\n'.join(headers)
	s = re.sub("[\\n]{2,}", "\n", s)
	return s


# def getIncludes(header):
# 	lines = header.split('\n')
# 	includes = []
# 	for line in lines:
# 		if line.startswith("#include"):
# 			includes.append(line)
# 	return includes

def getStructs(header):
	return re.findall(r"(typedef[\s]+struct[\s]*[\w]*[\s]*{[^}]+}[\s]*[\*\w,\s]+);", header)


def getFields(structure):
	# remover definição
	structure = structure.replace('\n', ' ')
	strc = {'name_t': None, 'name': None, 'atributte': [], 'methods': []}
	declaracao = structure.split('{')[0]
	endDeclaracao = structure.split('}')[-1]
	body = re.sub(declaracao + '[\s\\{]*', '', structure)
	body = re.sub('[\s\\}]*' + endDeclaracao.replace('*', '\\*'), '', body)
	for s in declaracao.split(' '):
		if s not in ['struct', 'typedef', '']: break
	strc['name_t'] = s
	for s in endDeclaracao.replace('*', '').split(','):
		s = s.replace(' ', '')
		if s not in ['']: break
	strc['name'] = s
	body = re.sub('[\\s]*;[\\s]*', ';', body)
	body = [b for b in body.split(';') if b != '']
	for field in body:
		name = re.findall('\\(\\*[\\w]+\\)', field)
		if len(name) > 0:  # is a method
			# <return type>(*<name>)(<args>)
			method = {}
			method['name'] = name[0][2:-1]
			method['rtype'] = field.split('(')[0].strip()
			i = 0
			while True:
				if field[i] == ')':
					break
				i += 1
			i += 1
			params = field[i:][1:-1].split(',')
			for i in range(len(params)):
				p = params[i]
				param = {}
				if p == 'struct %s *self' % (strc['name_t'],):
					param['type'] = strc['name']
					param['name'] = 'selfp'
				elif p == 'struct %s **selfpp' % (strc['name_t'],):
					param['type'] = strc['name'] + ' *'
					param['name'] = 'selfp'
				else:
					param['type'] = re.sub("[\\w]+$", '', p).strip()
					param['name'] = re.findall("[\\w]+$", p)[-1].strip()
				params[i] = param
			method['params'] = params
			strc['methods'].append(method)
		# print(params)
		else:
			# <type> <name>
			arg = {}
			arg['name'] = re.findall("[\\w]+$", field)[-1].strip()
			arg['type'] = re.sub("[\\w]+$", '', field).strip()
			strc['atributte'].append(arg)
	return strc


# print(strc['args'], sep='\n')
_TYPES_ = {
	'const char *': 'ctypes.c_char_p',
	'char *': 'ctypes.c_char_p',
	'size_t': 'ctypes.c_uint64',
	'void *': 'ctypes.c_void_p',
	'int8_t': 'ctypes.c_int8',
	'int': 'ctypes.c_int32',
	'REAL': 'ctypes.c_float',
	'void': 'None',
	'P3d': 'P3D',
	'P2d': 'P2D',
	'RandomParams': 'RDP',
	'Parametros': 'Params',
	'RdParams': 'RDP',
	'const P3d': 'P3D',
}
_MTYPES_ = {
	'const char *': 'ctypes.c_char_p',
	'char *': 'ctypes.POINTER(ctypes.c_char)',
	'size_t': 'ctypes.c_uint64',
	'void *': 'ctypes.POINTER(ctypes.c_void_p)',
	'int8_t': 'ctypes.c_int8',
	'int': 'ctypes.c_int32',
	'REAL': 'ctypes.c_float',
	'void': 'None',
	'P3d': 'P3d',
	'Parametros': 'Params',
	'const P3d': 'P3D',
	'P2d': 'P2D',
	'RandomParams': 'RDP',
	'RdParams': 'RDP',
}


def CTYPE(ctype):
	if ctype in _TYPES_:
		return _TYPES_[ctype]
	return 'ctypes.c_void_p'


def MCTYPE(ctype):
	if ctype in _MTYPES_:
		return _MTYPES_[ctype]
	return 'ctypes.c_void_p'


def makeParams(params):
	s = ','.join([CTYPE(p['type']) for p in params])
	if s != '':
		return ', ' + s
	return ''


# print(structure)
def createPyClass(Cclass, fileclass=sys.stdout):
	def mprint(*args, **kwargs):
		print(*args, **kwargs, file=fileclass)

	mprint(
		f"""import ctypes

class RDP(ctypes.Structure):
	_fields_ = [('typpe', ctypes.c_int32), ('a', ctypes.c_float), ('b', ctypes.c_float)]

	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

class Params(ctypes.Structure):
	_fields_ = [('hitlearn', ctypes.c_float), ('momento', ctypes.c_float), ('decaimento', ctypes.c_float), ('learnable', ctypes.c_int32) ]	
	
	def __init__(self,  *args, **kw):
		super().__init__(*args, **kw)
		

class P2D(ctypes.Structure):
	_fields_ = [('x', ctypes.c_uint64), ('y', ctypes.c_uint64)]
	
	def __init__(self,  *args, **kw):
		super().__init__(*args, **kw)


class P3D(ctypes.Structure):
	_fields_ = [('x', ctypes.c_uint64), ('y', ctypes.c_uint64), ('z', ctypes.c_uint64)]
	
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)


class {Cclass['name_t']}(ctypes.Structure):""", )
	att = []
	for attb in Cclass['atributte']:
		mprint("\t%s: %s" % (attb['name'], CTYPE(attb['type'])))
		att.append("('%s', %s)" % (attb['name'], CTYPE(attb['type'])))
	for mth in Cclass['methods']:
		n = 'cdll_' + mth['name']
		m = 'ctypes.CFUNCTYPE(%s%s)' % (MCTYPE(mth['rtype']), makeParams(mth['params']))
		mprint("\t%s: %s" % (n, m))
		att.append("('%s', %s)" % (n, m))  # mprint(mth)

	# ctypes.CFUNCTYPE(rest,arg)
	mprint("\t_fields_ = [", end='')
	mprint(*att, sep=',\n\t\t\t', end=']\n')
	mprint("\t@property\n"
		   "\tdef selfp(self):\n"
		   "\t\treturn  ctypes.cast(ctypes.addressof(self),ctypes.POINTER(%s))\n" % (Cclass['name_t'],))

	mprint("\t@property\n"
		   "\tdef selfpp(self):\n"
		   "\t\treturn ctypes.addressof(self.selfp)")

	# for mth in Cclass['methods']:
	# 	mprint()
	# 	mprint("\tdef %s(self, %s) -> %s:" % (mth['name'], ', '.join([p['name'] for p in mth['params'] if p['name'] != 'selfp']), CTYPE(mth['rtype'])))
	# 	mprint("\t\trt_value = self.%s(%s)" % ('cdll_' + mth['name'], ', '.join(['self.' + p['name'] if 'self' in p['name'] else p['name'] for p in mth['params']])))
	# 	mprint("\t\treturn rt_value")

	mprint(f"""
gab_dll = ctypes.CDLL(r"D:\\Henrique\\Rede-convolucional\\Gab\\bin\\libgab_cnn.dll")

gab_dll.{Cclass['name']}_new.restype = ctypes.POINTER(ctypes.c_void_p)

gab_dll.gab_realloc.restype = ctypes.POINTER(ctypes.c_void_p)
gab_dll.gab_realloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint64]
class {Cclass['name']}(ctypes.c_void_p):
""")
	for attb in Cclass['atributte']:
		mprint("\t%s: %s" % (attb['name'], CTYPE(attb['type'])))
	mprint(f"""
	def __init__(self, *args, **kw):
		p = gab_dll.Cnn_new()
		ctypes.memmove(ctypes.addressof(self), ctypes.addressof(p), 8)
		self.py_reference = ctypes.cast(p, ctypes.POINTER(Cnn_t))

	def __getattribute__(self, item):
		try:
			return object.__getattribute__(self, item)
		except:
			return object.__getattribute__(self.py_reference[0], item)

	def __getitem__(self, item):
		try:
			return object.__getitem__(self, item)
		except:
			return object.__getitem__(self.py_reference[0], item)

	def __del__(self):
		self.cdll_release(self)
		pass
	""")
	for mth in Cclass['methods']:
		mprint()
		mprint("\tdef %s(self, %s) -> %s:" % (mth['name'], ', '.join([p['name'] for p in mth['params'] if p['name'] != 'selfp']), MCTYPE(mth['rtype'])))
		mprint("\t\trt_value = self.%s(%s)" % ('cdll_' + mth['name'], ', '.join(['self' if 'self' in p['name'] else p['name'] for p in mth['params']])))
		if MCTYPE(mth['rtype']) == 'ctypes.POINTER(ctypes.c_char)':
			mprint("\t\tv = rt_value")
			mprint("\t\trt_value = ctypes.cast(v,ctypes.c_char_p).value.decode('utf-8')")
			mprint("\t\tgab_dll.gab_free(v)")
		mprint("\t\treturn rt_value")


def generateWrapper(file):
	header = loadFile(file)
	header = removeComments(header)
	header = removeSpaces(header)
	# includes = getIncludes(header)
	structs = getStructs(header)
	# print(header, file=open("text.txt", "w"))

	strc = getFields(structs[0])
	# print(*strc['methods'],sep='\n\n')
	createPyClass(strc, open('cnn.py', 'w'))


generateWrapper('include/cnn/cnn.h')
