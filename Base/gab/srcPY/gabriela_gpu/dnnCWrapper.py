import ctypes as c
import os
from platform import architecture

clib = None
__dir = ''
if architecture()[0] == '64bit':
    temp = os.path.abspath(__file__)
    temp = os.path.realpath(temp)
    temp = os.path.dirname(temp)
    # temp = os.path.join(temp, "lib/gabriela6.dll")
    __dir = temp
    temp = os.path.join(temp, "lib/libgab.dll")
    clib = c.CDLL(temp)
else:
    raise Exception('unsuport 32 bits architecture ')


class c_Gab(c.Structure):
    _fields_ = [("gab", c.c_void_p), ("size", c.c_uint)]


# void teste();

# int create_DNN(Gab *p_gab, int *arq, int l_arq, double hitLean);
clib.create_DNN.argtypes = [c.c_void_p, c.c_void_p, c.c_int, c.c_void_p, c.c_void_p, c.c_double]

# void initGPU(const char *src);
clib.initGPU.argtypes = [c.c_char_p]

# void initWithFile(const char *filename);
clib.initWithFile.argtypes = [c.c_char_p]

# void endGPU();

# void call(Gab *p_gab, double *inp);
clib.call.argtypes = [c.c_void_p, c.c_void_p]

# void learn(Gab *p_gab, double *trueOut);
clib.learn.argtypes = [c.c_void_p, c.c_void_p]

# void release(Gab *p_gab);
clib.release.argtypes = [c.c_void_p]

# void getoutput(Gab *p_gab, double *out);
clib.getoutput.argtypes = [c.c_void_p, c.c_void_p]

clib.sethitlearn.argtypes = [c.c_void_p, c.c_double]
clib.getA.argtypes = [c.c_void_p, c.c_int, c.c_void_p]

clib.setSeed.argtypes = [c.c_ulong]

clib.randomize.argtypes = [c.c_void_p]


class __Handle_gpu():
    def __init__(self, file: str):
        print("init gpu")
        clib.initWithFile(c.c_char_p(file.encode('utf-8')))

    def __del__(self):
        print("end gpu")
        clib.endGPU()


__handle_gpu = __Handle_gpu(os.path.join(__dir, "lib/gpu_kernels.cl"))


def setSeed(seed):
    clib.setSeed(seed)


def teste():
    clib.test()
