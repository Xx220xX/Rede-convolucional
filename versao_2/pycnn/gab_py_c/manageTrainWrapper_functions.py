from cnn_wrapper_structs import *
clib.manage2WorkDir.argtypes = [c.c_void_p]
clib.manage2WorkDir.restype = None
clib.releaseManageTrain.argtypes = [c.c_void_p]
clib.releaseManageTrain.restype = None
clib.manageTrainSetEvent.argtypes = [c.c_void_p,c.c_void_p]
clib.manageTrainSetEvent.restype = None
clib.manageTrainSetRun.argtypes = [c.c_void_p,c.c_int]
clib.manageTrainSetRun.restype = None
clib.createManageTrain.argtypes = [c.c_void_p,c.c_double,c.c_double,c.c_double]
clib.createManageTrain.restype = c.c_void_p
clib.ManageTrainloadImages.argtypes = [c.c_void_p]
clib.ManageTrainloadImages.restype = c.c_int
clib.ManageTraintrain.argtypes = [c.c_void_p]
clib.ManageTraintrain.restype = c.c_int
clib.ManageTrainfitnes.argtypes = [c.c_void_p]
clib.ManageTrainfitnes.restype = c.c_int
clib.manageTrainLoop.argtypes = [c.c_void_p,c.c_int]
clib.manageTrainLoop.restype = None