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
            'cl_context':'{ctypes}.c_void_p',
            'QUEUE':'{ctypes}.c_void_p',
            'UINT':'{ctypes}.c_void_p',
            'cl_mem':'{ctypes}.c_void_p',
            'cl_kernel':'{ctypes}.c_void_p',
            'cl_int':'{ctypes}.c_int',
            'cl_ulong':'{ctypes}.c_uint64',
            'Typecamada':'Camada',
            'Tensor':'TOPOINTER(Tensor)',
            'TensorChar':'TOPOINTER(Tensor)',
            }

CPYTYPES_P = {
            'char': '{ctypes}.c_char_p',
            'void': '{ctypes}.c_void_p',
            'wchar':'{ctypes}.c_wchar_p',
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