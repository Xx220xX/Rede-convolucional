PathOut = '../../pyTrain/gab_py_c/'

file_load_dll = 'gab_wrapper_load_dll'
file_structs = 'gab_wrapper_structs'
file_class = 'gab_wrapper_class'
file_functions = 'gab_wrapper_functions'
file_load = 'gab_wrapper'
dll_file_name = 'libCNNGPU.dll'

ctypes_name = 'c'
dll_var_name = 'clib'

pathInclude_h = '../../CNN_GPU/include/cnn/'
h_structs = [
	"utils/String.h",
	"utils/list_args.h",
	"tensor/Tensor.h",
	"gpu/Kernel.h",

	"camadas/Camada.h",
	"camadas/CamadaBatchNorm.h",
	"camadas/CamadaConv.h",
	"camadas/CamadaConvNC.h",
	"camadas/CamadaDropOut.h",
	"camadas/CamadaFullConnect.h",
	"camadas/CamadaPadding.h",
	"camadas/CamadaPool.h",
	"camadas/CamadaPoolAv.h",
	"camadas/CamadaRelu.h",
	"camadas/CamadaSoftMax.h",
	'cnn.h',
	"utils/manageTrain.h"
]

h_functions = [
	"tensor/Tensor.h",
	'cnn.h',
	"utils/manageTrain.h",
	"libraryPythonWrapper.h"
]

# gerar leitura da dll
print('gerar leitura da dll')
file = open(PathOut + file_load_dll + '.py', 'w')

print(
	f"""
import ctypes as {ctypes_name}
import os
__dir = os.path.abspath(__file__)
__dir = os.path.realpath(__dir)
__dir = os.path.dirname(__dir)
__dll = os.path.join(__dir, '../bin/{dll_file_name}')

{dll_var_name} = {ctypes_name}.CDLL(__dll)
""", file=file)
file.close()

# gerar arquivo de estruturas
# este nao depende da DLL
print('gerar arquivo de estruturas')
from generateClass import *

file = open(PathOut + file_structs + '.py', 'w')

print(f"import ctypes as {ctypes_name}", file=file)
print("""
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
    """, file=file)

for h in h_structs:
	putClassInFile(pathInclude_h + h, file)

file.close()

# Gerar wrapper de funções
print('Gerar wrapper de funções')
file = open(PathOut + file_functions + '.py', 'w')
print(
	f'''
try:
	from {file_class}  import *
except Exception:
	from gab_py_c.{file_class}  import *
	
''', file=file)

from generateFunctions import *

for h in h_functions:
	putFunctionInFile(pathInclude_h + h, file, dll_var_name, ctypes_name)

file.close()
# input('finalizado')
