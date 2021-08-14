import ctypes as c

import numpy as np
from math import *
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


class String(c.Structure):
	_fields_ = [
		('d', c.c_char_p),
		('size', c.c_uint64),
		('release', c.c_char),
	]


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
		('tr_time', c.c_size_t),
		('ft_imagem_atual', c.c_uint),
		('ft_numero_imagens', c.c_uint),
		('ft_info', c.POINTER(c.c_double)),
		('ft_numero_classes', c.c_uint),
		('ft_time', c.c_size_t),
	]


class ManageTrain(c.Structure):
	_fields_ = [
		('et', Estatistica),
		('cnn', c.c_void_p),
		('homePath', String),
		('file_images', String),
		('file_labels', String),
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
		('class_names', String),
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
		('UpdateTrain', ManageEvent),
		('UpdateFitnes', ManageEvent),
		('self_release', c.c_char),
		('real_time', c.c_char),
		('process', c.c_void_p),
		('update_loop', c.c_void_p),
		('exist', c.c_void_p),
		('can_run', c.c_uint),
		('process_id', c.c_uint),
	]


global_act: Activity.Activity
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
	act.setVar('epoca', 'Epoca 1')
	act.setVar('pg_epoca', 0)


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnfinishEpic(t: c.POINTER(ManageTrain)):
	et: Estatistica
	act: Activity.Activity
	global global_act
	act = global_act
	t = t[0]
	et = t.et
	act.setVar('epoca', 'Epoca %d' % (et.tr_epoca_atual + 2))
	act.setVar('pg_epoca', et.tr_epoca_atual + et.tr_imagem_atual / et.tr_numero_imagens)


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def OnInitTrain(t: c.POINTER(ManageTrain)):
	global global_act
	act = global_act
	t = t[0]
	et = t.et


def gettime(seg):
	seg = round(seg)
	t = []
	dia = floor(seg / 86400)
	seg = seg % 86400
	hora = floor(seg / 3600)
	seg = seg % 3600
	minuto = floor(seg / 60)
	seg = seg % 60
	if dia > 0:
		t.append('%d d' % (dia,))
	if hora > 0:
		t.append('%2d h' % (hora,))
	if minuto > 0:
		t.append('%2d m' % (minuto,))
	if dia > 0:
		t.append('%2d s' % (seg,))
	t = ' '.join(t)
	if t == '':
		t = '0 s'
	return t


@c.CFUNCTYPE(c.c_void_p, c.c_void_p)
def UpdateTrain(t: c.POINTER(ManageTrain)):
	global global_act
	act = global_act
	t = t[0]
	et = t.et
	imagem_progress = (et.tr_imagem_atual + 1) / et.tr_numero_imagens
	act.setVar('epoca', 'Epoca %d' % (et.tr_epoca_atual+1,))
	act.setVar('imagem', 'Imagem %d' % (et.tr_imagem_atual+1,))
	act.setVar('pg_imagem', imagem_progress)

	imagem_por_segundo = (et.tr_epoca_atual * et.tr_numero_imagens + et.tr_imagem_atual) / et.tr_time * 1000
	tempo_fim_treino = ((
								et.tr_numero_epocas - et.tr_epoca_atual) * et.tr_numero_imagens - et.tr_imagem_atual - 1) / imagem_por_segundo
	tempo_fim_epoca = (et.tr_numero_imagens - et.tr_imagem_atual - 1) / imagem_por_segundo

	if imagem_por_segundo <= 0:
		act.setVar('timagem', 'calculando')
		act.setVar('tepoca', 'calculando')
	else:
		act.setVar('timagem', gettime(tempo_fim_epoca))
		act.setVar('tepoca', gettime(tempo_fim_treino))
	act.setVar('mse',float(et.tr_erro_medio))
	act.setVar('win_rate',float(et.tr_acerto_medio))
	length = et.tr_imagem_atual
	x = [float(et.tr_mse_vector[v]) for v in range(length)]
	y = [float(et.tr_acertos_vector[v]) for v in range(length)]
	end = et.tr_epoca_atual+et.tr_imagem_atual/et.tr_numero_imagens
	ep = list(np.linspace(0,end,length))
	act.extras['plt0'](ep,x,ep,y)


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
