import ctypes as c

import numpy as np

import Activity


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


ManageEvent = c.CFUNCTYPE(c.c_void_p, c.c_void_p)


class Estatistica(c.Structure):
	_fields_ = [
		('tr_mse_vector', c.POINTER(c.c_double)),
		('tr_acertos_vector', c.POINTER(c.c_double)),
		('tr_imagem_atual', c.c_uint),
		('tr_numero_imagens', c.c_uint),
		('tr_epoca_atual', c.c_uint),
		('tr_numero_epocas', c.c_uint),
		('tr_erro_medio', c.c_double),
		('tr_acerto_medio', c.c_double),

		('ft_imagem_atual',c.c_uint),
		('ft_numero_imagens',c.c_uint),
		('ft_erro_medio',  c.POINTER(c.c_double)),
		('ft_acerto_medio',  c.POINTER(c.c_double)),
		('ft_numero_classes', c.c_uint)
	]


class ManageTrain(c.Structure):
	_fields_ = [
		('et', Estatistica),
		('cnn', c.c_void_p),
		('homePath', c.c_char_p),
		('file_images', c.c_char_p),
		('file_labels', c.c_char_p),
		('headers_images', c.c_uint),
		('headers_labels', c.c_uint),
		('imagens', c.c_void_p),
		('targets', c.c_void_p),
		('labels', c.c_void_p),
		('n_epics', c.c_int),
		('epic', c.c_int),
		('n_images', c.c_int),
		('n_images2train', c.c_int),
		('n_images2fitness', c.c_int),
		('image', c.c_int),
		('n_classes', c.c_int),
		('class_names', c.c_char_p),
		('character_sep', c.c_char),
		('sum_erro', c.c_double),
		('sum_acerto', c.c_int),
		('current_time', c.c_double),
		('OnloadedImages', ManageEvent),
		('OnfinishEpic', ManageEvent),
		('OnInitTrain', ManageEvent),
		('OnfinishTrain', ManageEvent),
		('OnInitFitnes', ManageEvent),
		('OnfinishFitnes', ManageEvent),
		('can_run', c.c_int),
		('process', c.c_void_p),
		('self_release', c.c_char),
		('releaseStrings', c.c_char),
	]


global_act = None


def setAct(act):
	global global_act
	global_act = act


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnloadedImages(t: c.POINTER(ManageTrain)):
	et: Estatistica
	act: Activity.Activity
	global global_act
	act = global_act
	t = t[0]
	et = t.et
	act.setVar('epoca','Epoca 1')
	act.setVar('pg_epoca',0)


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnfinishEpic(t: c.POINTER(ManageTrain)):
	et: Estatistica
	act: Activity.Activity
	global global_act
	act = global_act
	t = t[0]
	et = t.et
	act.setVar('epoca','Epoca %d'%(et.tr_epoca_atual+2))
	act.setVar('pg_epoca',et.tr_epoca_atual + et.tr_imagem_atual/et.tr_numero_imagens)




@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnInitTrain(t: c.POINTER(ManageTrain)):
	global global_act
	act = global_act
	t = t[0]


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnfinishTrain(t: c.POINTER(ManageTrain)):
	global global_act
	act = global_act
	t = t[0]


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnInitFitnes(t: c.POINTER(ManageTrain)):
	global global_act
	act = global_act
	t = t[0]


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnfinishFitnes(t: c.POINTER(ManageTrain)):
	global global_act
	act = global_act
	t = t[0]
