import re
import pytypes
import os


def removeComents(cprogram):
	s = re.sub(r"((\/\*)[?=\*]*[^(\*\/)]*\*\/)", "", cprogram)
	s = re.sub(r"(\/\/[^\r\n]*)", "", s)
	return s


def detectStruct(msg):
	return re.findall(r"(typedef[\s]+struct[\s]*[\w]*[\s]*{[^}]+}[\s]*[\*\w,\s]+);", msg)


def structGetName(stc):
	title = re.findall(r'(struct[\s\w]*)', stc)
	bottom = re.findall(r'(}[\s\w,\*]+)', stc)
	title = title[0].replace('struct', '').replace(' ', '')
	bottom = re.findall(r'([\w]+)', bottom[0])
	nomes = [title] + bottom
	nomes = [_ for _ in nomes if _ != '']
	if title == '':
		return [nomes[0]]
	if len(nomes) > 1:
		return [nomes[1]]
	return nomes


def structGetvars(s):
	s = re.findall(r'(\{[^\{\}]+\})', s)[0][1:-1]
	s = re.sub(r'[\s]+', ' ', s)
	s = re.findall(r'([\w]+[\s]*[,\*\w\s]+)', s)
	vars = []
	for declaracao in s:
		seq = re.split(r'(^[\w]+)', declaracao)
		tipo = seq[1]
		nomes = seq[2].replace(' ', '').split(',')
		for nome in nomes:
			n = re.findall('\*', nome)
			n = len(n)
			desc = (tipo + '*' * n, nome.replace('*', ''))
			vars.append(desc)
	return vars


def cvarToPy(variables: list):
	pvars = []
	for v in variables:
		tipo, nome = v[0], v[1]
		t = tipo.split('*')
		tp = t.pop(0)
		for _ in t:
			tp = pytypes.toPointer(tp)
		tp = pytypes.getPyType(tp)
		pvars.append((nome, tp))
	return pvars


# outfile = open('../gab_py_c/cnn_wrapper_structs.py', 'w')

def putClassInFile(file_h, fpy, ctypes_name='c'):
	c_h = removeComents(open(file_h, 'r').read())
	stcs = detectStruct(c_h)
	for structs in stcs:
		nomes = structGetName(structs)
		variables = structGetvars(structs)
		variablespy = cvarToPy(variables)

		# print(f"class {nomes[0]}({ctypes_name}.Structure):\n\tpass", file=open('tmp.py','a'))
		print(f"class {nomes[0]}({nomes[0]}):", file=fpy)
		for v in variablespy:
			print(f"\t{v[0]}: {v[1].replace('{ctypes}', ctypes_name)}", file=fpy)
		print("\t_fields_ = [", file=fpy)
		for v in variablespy:
			print(f"\t\t('{v[0]}', {v[1].replace('{ctypes}', ctypes_name)}),", file=fpy)
		print("\t]", file=fpy)
		# for n in nomes[1:]:
		#     print(f"{n} = {nomes[0]}", file=fpy)
		print('\n', file=fpy)


def putClassInString(file_h, string_class: str, ctypes_name='c'):
	c_h = removeComents(open(file_h, 'r').read())
	stcs = detectStruct(c_h)
	print('\tFile',file_h)
	for structs in stcs:
		nomes = structGetName(structs)
		variables = structGetvars(structs)
		variablespy = cvarToPy(variables)
		print('\t\tClass',nomes[0])
		# print(f"class {nomes[0]}({ctypes_name}.Structure):\n\tpass", file=open('tmp.py','a'))
		# print(f"class {nomes[0]}({nomes[0]}):", file=fpy)
		local_str = []
		for v in variablespy:
			local_str.append(f"{v[0]}: {v[1].replace('{ctypes}', ctypes_name)}")
		local_str.append("_fields_ = [")
		for v in variablespy:
			local_str.append(f"\t('{v[0]}', {v[1].replace('{ctypes}', ctypes_name)}),")
		local_str.append("]")

		# for n in nomes[1:]:
		#     print(f"{n} = {nomes[0]}", file=fpy)
		local_str.append('')
		local_str = '\n\t'.join(local_str)
		string_class = string_class.replace('# ${%s}'%(nomes[0],), local_str)
	return string_class

if __name__ == '__main__':
	outfile = None
	ctypes_name = 'c'

	path_include = r"D:/Henrique/Rede-convolucional/CNN_GPU/include/cnn/"

	camadas = [path_include + 'tensor/Tensor.h']

	# gg = [path_include + 'gpu/Kernel.h'] + \
	#           [path_include+'tensor/Tensor.h'] + \
	#           [path_include + 'camadas/'+ camada for camada in
	#            os.listdir(path_include+ 'camadas/') if camada.endswith('.h')]

	print("""
    from wrapper_dll import *
    EXCEPTION_MAX_MSG_SIZE = 500
    def TOPOINTER(c_type):
        tp = c.POINTER(c_type)
        def get(self, item):
            return self[0].__getattribute__(item)
        def set(self, key, value):
            self[0].__setattr__(key, value)
        def rep(self):
            return self[0].__repr__()
        tp.__getattribute__ = get
        tp.__setattr__ = set
        tp.__repr__ = rep
        return tp
    """, file=outfile)

	for file in camadas:
		putClassInFile(file, outfile)

	print('finalizado')
