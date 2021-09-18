# import ctypes as c
# import inspect
# for n,o in inspect.getmembers(c):
#     if inspect.isclass(o):
#         print(f"'{n.replace('c_','')}':'{{ctypes}}.{n}',")
#
import ctypes

CPYTYPES = {'bool': '{ctypes}.c_bool',
			'byte': '{ctypes}.c_byte',
			'char': '{ctypes}.c_char',
			'char_p': '{ctypes}.c_char_p',
			'double': '{ctypes}.c_double',
			'float': '{ctypes}.c_float',
			'int': '{ctypes}.c_int',
			'int16': '{ctypes}.c_int16',
			'int32': '{ctypes}.c_int32',
			'int64': '{ctypes}.c_int64',
			'int8': '{ctypes}.c_int8',
			'long': '{ctypes}.c_long',
			'longdouble': '{ctypes}.c_longdouble',
			'longlong': '{ctypes}.c_longlong',
			'short': '{ctypes}.c_short',
			'size_t': '{ctypes}.c_size_t',
			'ssize_t': '{ctypes}.c_ssize_t',
			'ubyte': '{ctypes}.c_ubyte',
			'uint': '{ctypes}.c_uint',
			'uint16': '{ctypes}.c_uint16',
			'uint32': '{ctypes}.c_uint32',
			'uint64': '{ctypes}.c_uint64',
			'uint8': '{ctypes}.c_uint8',
			'ulong': '{ctypes}.c_ulong',
			'ulonglong': '{ctypes}.c_ulonglong',
			'ushort': '{ctypes}.c_ushort',
			'void_p': '{ctypes}.c_void_p',
			'voidp': '{ctypes}.c_voidp',
			'wchar': '{ctypes}.c_wchar',
			'wchar_p': '{ctypes}.c_wchar_p',
			'f2v': '{ctypes}.CFUNCTYPE({ctypes}.c_int,{ctypes}.c_void_p,{ctypes}.c_void_p)',
			'fv': '{ctypes}.CFUNCTYPE({ctypes}.c_int,{ctypes}.c_void_p)',
			'fvc': '{ctypes}.CFUNCTYPE({ctypes}.c_int,{ctypes}.c_void_p,{ctypes}.c_char)',
			'fv3d': '{ctypes}.CFUNCTYPE({ctypes}.c_int,{ctypes}.c_void_p,{ctypes}.c_double,{ctypes}.c_double,{ctypes}.c_double)',
			'f4v': '{ctypes}.CFUNCTYPE({ctypes}.c_int,{ctypes}.c_void_p,{ctypes}.c_void_p,{ctypes}.c_void_p,{ctypes}.c_void_p)',
			'cfv': '{ctypes}.CFUNCTYPE({ctypes}.c_char_p,{ctypes}.c_void_p)',
			'ManageEvent': '{ctypes}.CFUNCTYPE(None,{ctypes}.c_void_p)',
			'cl_context': '{ctypes}.c_void_p',
			'QUEUE': '{ctypes}.c_void_p',
			'UINT': '{ctypes}.c_uint',
			'cl_mem': '{ctypes}.c_void_p',
			'cl_kernel': '{ctypes}.c_void_p',
			'cl_int': '{ctypes}.c_int',
			'cl_ulong': '{ctypes}.c_uint64',
			'flag_t': '{ctypes}.c_uint8',
			'Typecamada': 'Camada',
			'Tensor': 'TOPOINTER(Tensor)',
			'TensorChar': 'TOPOINTER(Tensor)',
			'Kernel': '{ctypes}.c_void_p',
			'Thread': '{ctypes}.c_void_p',
			'atomic_int': '{ctypes}.c_uint',
			'pthread_cond_t': '{ctypes}.c_void_p',
			'Ptr': '{ctypes}.c_void_p',
			'char_500': '{ctypes}.c_char_p*EXCEPTION_MAX_MSG_SIZE',
			# 'Kernel':'TOPOINTER(Kernel)',
			}

CPYTYPES_P = {
	'char': '{ctypes}.c_char_p',
	'void': '{ctypes}.c_void_p',
	'wchar': '{ctypes}.c_wchar_p',
	'WrapperCL': '{ctypes}.c_void_p',
	'pthread_t': '{ctypes}.c_void_p',
	'Camada': 'c.POINTER(TOPOINTER(Camada))',


}

CPYFUNCTYPES = {
	'Cnn': '{ctype}.c_void_p',
	'WrapperCL *': '{ctype}.c_void_p',
	'Params': 'Params',
	'UINT': '{ctype}.c_uint32',
	'void': 'None',
	'Cnn *': '{ctype}.c_void_p',
	'const char *': '{ctype}.c_void_p',
	'unsigned long long int': '{ctype}.c_uint64',
	'ULL': '{ctype}.c_uint64',
	'int': '{ctype}.c_int',
	'double *': '{ctype}.c_void_p',
	'char': '{ctype}.c_char',
	'double': '{ctype}.c_double',
	'long long int': '{ctype}.c_int64',
	'void *': '{ctype}.c_void_p',
	'FILE *': '{ctype}.c_void_p',
	'char *': '{ctype}.c_void_p',
	'size_t *': '{ctype}.c_void_p',
	'String *': '{ctype}.c_void_p',
	'Pointer *': '{ctype}.c_void_p',
	'Camada': '{ctype}.c_void_p',
	'Tensor': '{ctype}.c_void_p',
	'cl_context': '{ctype}.c_void_p',
	'QUEUE': '{ctype}.c_void_p',
	'CNN_ERROR *': '{ctype}.c_void_p',
	'size_t': '{ctype}.c_uint64',
	'UINT *': '{ctype}.c_void_p',
	'Tensor *': '{ctype}.c_void_p',
	'Kernel': '{ctype}.c_void_p',
	'ManageTrain *': '{ctype}.c_void_p',
	'ManageTrain': '{ctype}.c_void_p',
	'ManageEvent *': '{ctype}.c_void_p',
	'ManageEvent': '{ctype}.c_void_p',
	'cl_command_queue': '{ctype}.c_void_p',
	'unsigned int': '{ctype}.c_uint',
	'uint64_t': '{ctype}.c_uint64',
	'RandomParam': 'RandomParam',

}


def toPointer(ctp):
	if ctp in CPYTYPES_P:
		return CPYTYPES_P[ctp]
	if ctp in CPYTYPES:
		ctp = CPYTYPES[ctp]
	return f"{{ctypes}}.POINTER({ctp})"


def getPyType(ctp):
	if ctp in CPYTYPES:
		return CPYTYPES[ctp]
	return ctp
