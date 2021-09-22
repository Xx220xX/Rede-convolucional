import re
import pytypes
import os
from pytypes import CPYFUNCTYPES


def removeComents(cprogram):
	s = re.sub(r"((\/\*)[?=\*]*[^(\*\/)]*\*\/)", "", cprogram)
	s = re.sub(r"((\/\*)[?=\*]*[^(\/)]*\/)", "", s)
	s = re.sub(r"(\/\/[^\r\n]*)", "", s)
	return s


def removePrecptor(cprogram):
	s = re.sub(r"(#[^\r\n]*)", "", cprogram)
	return s


def removeTypedef(cprogram):
	s = re.sub(r"(typedef[^}]*}[\w\*,\s]*;)", "", cprogram)
	return s


def removeExcessSpaces(cprogram):
	s = re.sub(r"([\s]{2,})", " ", cprogram)
	return s


def putLines(cprogram):
	s = re.sub(r";[\s]*", ";\n", cprogram)
	return s


def getFunctions(cprogram: str):
	return cprogram.split('\n')


def getFunctionPrototype(func: str):
	if func == '' or func == ' ': return None
	returnType = re.sub('[\s]*[\w_]+[\s]*\([^;]*;', "", func)
	name = re.sub('^([const ]*[\w]+\s[\*]*)', "", func)
	args = (re.findall("\([^;]+", func))[0]
	name = re.sub('\([^;]+;', "", name)
	args = args.replace('(', '').replace(')', '')
	args = args.split(',')
	ag = []
	for a in args:
		v = re.sub('[\s]*[\w]+$', '', a).strip()
		if v != "":
			ag.append(v)
	args = ag
	return returnType.strip(), name.strip(), args


def getProtoFromCode(code):
	code = removeComents(code)
	code = removePrecptor(code)
	code = removeTypedef(code)

	code = removeExcessSpaces(code)

	code = putLines(code)
	funcs = getFunctions(code)
	# print('\n'.join(funcs))
	# input()
	proto = []
	for f in funcs:
		if 'extern ' in f:
			continue
		ff = getFunctionPrototype(f)
		if ff is None:
			continue
		proto.append(ff)

	return proto


def putFunctionInFile(hfile, outFile,filefuncs=None, clib='clib', ctypes='c'):
	proto = getProtoFromCode(open(hfile, 'r').read())
	for retorno, nome, argumentos in proto:
		ag = []
		for a in argumentos:
			ag.append(CPYFUNCTYPES[a])

		print(f'{clib}.{nome}.argtypes = [{",".join(ag)}]'.replace("{ctype}", ctypes), file=outFile)
		print(f'{clib}.{nome}.restype = {CPYFUNCTYPES[retorno]}'.replace("{ctype}", ctypes), file=outFile)
		if filefuncs:
			print(f'\tdef {nome}({", ".join(["self"]+["v%d"%(_,)for _ in range(len(ag))])}): pass\n', file = filefuncs)


if __name__ == '__main__':
	path_include = "../../CNN_GPU/include/cnn/"
	code = [
		# path_include + 'cnn.h',
		# path_include + 'libraryPythonWrapper.h',
		# path_include + 'tensor/Tensor.h'
		path_include + 'utils/manageTrain.h'
	]
	proto = []
	for fl in code:
		proto += getProtoFromCode(open(fl, 'r').read())

	fout = open('../gab_py_c/manageTrainWrapper_functions.py', 'w')
	lprint = print

	clib = 'clib'
	ctyp = 'c'


	def print(*args, file=fout, **kw):
		return lprint(*args, file=file, **kw)


	print("from cnn_wrapper_structs import *")

	for retorno, nome, argumentos in proto:
		ag = []
		for a in argumentos:
			ag.append(CPYFUNCTYPES[a])
		print(f'{clib}.{nome}.argtypes = [{",".join(ag)}]'.replace("{ctype}", ctyp))
		print(f'{clib}.{nome}.restype = {CPYFUNCTYPES[retorno]}'.replace("{ctype}", ctyp))
	fout.close()

	lprint('finalizado')
