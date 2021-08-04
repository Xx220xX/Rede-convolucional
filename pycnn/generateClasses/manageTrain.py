
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

class Estatistica(c.Structure):
	_fields_ = [
		('erros',c.POINTER(c.c_double)),
		('acertos',c.POINTER(c.c_double)),
		('fitness_hit_rate',TOPOINTER(Tensor)),
		('image_fitnes',c.c_uint),
		('max_size',c.c_uint),
		('image',c.c_uint),
		('epic',c.c_uint),
		('mean_error',c.c_double),
		('hit_rate',c.c_double),
	]
class ManageTrain(c.Structure):
	_fields_ = [
		('cnn',Cnn),
		('homePath',c.c_char_p),
		('file_images',c.c_char_p),
		('file_labels',c.c_char_p),
		('headers_images',c.c_uint),
		('headers_labels',c.c_uint),
		('imagens',TOPOINTER(Tensor)),
		('targets',TOPOINTER(Tensor)),
		('labels',TOPOINTER(Tensor)),
		('n_epics',c.c_int),
		('epic',c.c_int),
		('n_images',c.c_int),
		('n_images2train',c.c_int),
		('n_images2fitness',c.c_int),
		('image',c.c_int),
		('n_classes',c.c_int),
		('class_names',c.c_char_p),
		('character_sep',c.c_char),
		('sum_erro',c.c_double),
		('sum_acerto',c.c_int),
		('et',Estatistica),
		('current_time',c.c_double),
		('OnloadedImages',ManageEvent),
		('OnfinishEpic',ManageEvent),
		('OnInitTrain',ManageEvent),
		('OnfinishTrain',ManageEvent),
		('OnInitFitnes',ManageEvent),
		('OnfinishFitnes',ManageEvent),
		('can_run',atomic_int),
		('process',pid_t),
		('self_release',c.c_char),
		('releaseStrings',c.c_char),
	]
