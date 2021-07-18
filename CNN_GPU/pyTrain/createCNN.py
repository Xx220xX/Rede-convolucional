import ctypes as c
import tkinter as tk
import tkinter.font as Font

import copy

try:
    from Activity import *
except Exception:
    from pyTrain.Activity import *


class Arquitetura:
    def __init__(self):
        self.entrada = [1, 1, 1]
        self.camadas = []

    def update(self, entrada):
        self.entrada = entrada[:]
        sucess = True
        for camada in self.camadas:
            if sucess:
                sucess = camada['generateOut'](camada, entrada)
                entrada = camada['saida']
            else:
                camada['ok'] = False

    def add(self, layer):
        self.camadas.append(copy.deepcopy(layer))
    def remove(self,index):
        if index <0: index = len(self.camadas)+index
        del self.camadas[index]
def ui(act: Activity):
    act.text('Entrada', x=act.px(1), y=act.py(0.3), height=act.py(5))
    act.entry('x', 1, int, x=act.px(5), width=act.px(5), height=act.py(5), justify=tk.CENTER)
    act.entry('y', 1, int, x=act.px(10), width=act.px(5), height=act.py(5), justify=tk.CENTER)
    act.entry('z', 1, int, x=act.px(15), width=act.px(5), height=act.py(5), justify=tk.CENTER)
    fArch = act.Frame(act.frame, width=act.px(30), height=act.py(84), bg="#ff0")
    fEdit = act.Frame(act.frame, width=act.px(68), height=act.py(90), bg="#ff0")
    fSet = act.Frame(fEdit, width=fEdit.px(50), height=act.py(90), bg="#f0f")
    fAdd = act.Frame(act.frame, width=fEdit.w, height=act.py(89), bg="#ff0")

    lista = act.Listbox(frame=fArch, width=fArch.w, height=fArch.h,font=Font.Font(size=14))
    act.ScroolbarY(lista, frame=fArch, width=20, height=fArch.h)

    params = act.Listbox(frame=fEdit, x=21, y=act.py(6), width=fEdit.px(50), height=fEdit.py(94),font=Font.Font(size=16))
    act.ScroolbarY(params, frame=fEdit, y=act.py(6), width=20, height=fEdit.py(94))

    possibleLayers = act.Listbox(frame=fAdd, x=21, y=act.py(6), width=fAdd.px(50), height=fAdd.py(94),
                                 font=Font.Font(size=16))
    act.ScroolbarY(possibleLayers, frame=fAdd, y=act.py(6), width=20, height=fAdd.py(94))

    act.label('nomeCamada', "NULL", frame=fEdit, x=fEdit.px(1), y=act.py(0.3), height=act.py(5))
    act.button('oklayer', 'confirm', frame=fEdit, x=fEdit.px(85), y=act.py(0.3), width=fEdit.px(9), height=act.py(5))
    act.button('removeLayer', 'Remove', frame=fEdit, x=fEdit.px(75), y=act.py(0.3), width=fEdit.px(9), height=act.py(5))

    act.button('make', 'Compilar', frame=fAdd, x=fAdd.px(25), y=act.py(0.3), width=fAdd.px(23), height=act.py(5))

    act.extras['arch'] = Arquitetura()

    def updateArq(*args, **kw):
        print('here', *args)
        try:
            entrada = [act.getValue('x'), act.getValue('y'), act.getValue('z')]
        except Exception:
            return
        act.extras['arch'].update(entrada)
        lista.delete(0, tk.END)
        for camada in act.extras['arch'].camadas:
            lista.insert(tk.END, camada['name'])
            if not camada['ok']:
                lista.itemconfig(tk.END, {'fg': '#f00'})

    act.getEntry('x').bind('<Key>', updateArq)
    act.getEntry('y').bind('<Key>', updateArq)
    act.getEntry('z').bind('<Key>', updateArq)
    for layer in LAYERS:
        possibleLayers.insert(tk.END, layer['name'])
    possibleLayers.selection_set(0, 0)
    fArch.actplace(y=act.py(6))

    def addLayer(*args, **kw):
        select = possibleLayers.curselection()[0]
        act.extras['arch'].add(LAYERS[select])
        fAdd.place_forget()
        editLayer(act.extras['arch'].camadas[-1],-1)

    def editLayer(layer,index):
        act.setVar('nomeCamada', layer['name'])
        params.delete(0,tk.END)
        for arg in layer['args']:
            params.insert(tk.END,arg[0])
        fEdit.actplace(x=act.px(32))

    act.button('addLayer', 'Nova camada', frame=fAdd, x=fAdd.px(1), y=act.py(0.3), width=fAdd.px(23), height=act.py(5),
               command=addLayer)
    fEdit.actplace(x=act.px(32))
    fSet.actplace(x=fEdit.px(50)+20)
    #### continuar
    makeOptions(act,fSet)
    # fAdd.actplace(x=act.px(32))

def makeOptions(act:Activity,frame:tk.Frame):
    #2dim
    f2dim = act.Frame(frame,width=frame.w,height=frame.py(50),bg='#0f0')
    f2dim.actplace(y=frame.py(25))
    act.label('2dim_name','null',frame=f2dim,x=f2dim.px(25),width=f2dim.px(50))
    act.entry('2dim_x',1,int, frame=f2dim, y =30,x=f2dim.px(25),width=f2dim.px(50),justify=tk.CENTER)
    act.entry('2dim_y',1,int, frame=f2dim, y =60,x=f2dim.px(25),width=f2dim.px(50),justify=tk.CENTER)
    def update2dim(*args,**kw):
        try:
            x = act.getValue('2dim_x')
            y = act.getValue('2dim_y')
        except Exception:
            return
        act.extras['2dim_var'][2] = [x,y]
    act.getEntry('2dim_x').bind('<Key>', update2dim)
    act.getEntry('2dim_y').bind('<key>',update2dim)

import numpy as np


def checkconv(layer, entrada):
    im = np.array(entrada[:-1])
    f = np.array(layer['args'][2][2])
    p = np.array(layer['args'][1][2])
    s = (im - f) / p + 1
    layer['ok'] = False
    if s[0] != int(s[0]) or s[0] <= 0: return False
    if s[1] != int(s[1]) or s[1] <= 0: return False
    layer['saida'] = [s[0], s[1], layer['args'][3][2]]
    layer['ok'] = True
    return True


def checkconvNc(layer, entrada):
    im = np.array(entrada[:-1])
    p = np.array(layer['args'][1][2])
    a = np.array(layer['args'][2][2])
    f = np.array(layer['args'][3][2])
    s = (im - (f - 1) * a) / p + 1
    layer['ok'] = False
    if s[0] != int(s[0]) or s[0] <= 0: return False
    if s[1] != int(s[1]) or s[1] <= 0: return False
    layer['saida'] = [s[0], s[1], layer['args'][4][2]]
    layer['ok'] = True
    return True


def checkPool(layer, entrada):
    im = np.array(entrada[:-1])
    p = np.array(layer['args'][1][2])
    f = np.array(layer['args'][2][2])
    s = (im - f) / p + 1
    layer['ok'] = False
    if s[0] != int(s[0]) or s[0] <= 0: return False
    if s[1] != int(s[1]) or s[1] <= 0: return False
    layer['saida'] = [s[0], s[1], entrada[-1]]
    layer['ok'] = True
    return True


def checkPadding(layer, entrada):
    layer['saida'] = [entrada[0] + layer['args'][1][2] + layer['args'][2][2],
                      entrada[1] + layer['args'][3][2] + layer['args'][3][2],
                      entrada[2]
                      ]
    layer['ok'] = True
    return True


def checkGeneric(layer, entrada):
    layer['saida'] = entrada[:]
    layer['ok'] = True
    return True


def checkFull(layer, entrada):
    layer['saida'] = [layer['args'][1][2], 1, 1]
    layer['ok'] = True
    return True


LAYERS = [
    {'name': 'Convolucao', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('passo', '2dim', [1, 1]),
         ('filtro', '2dim', [1, 1]),
         ('número de filtros', 'int', 1),
         ('taxa de aprendizado', 'float', 0.0),
         ('momento', 'float', 0.0),
         ('decaimento de peso', 'float', 0.0)
     ], 'generateOut': checkconv, 'ok': False
     },
    {'name': 'ConvolucaoNcausal', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('passo', '2dim', [1, 1]),
         ('abertura', '2dim', [1, 1]),
         ('filtro', '2dim', [1, 1]),
         ('número de filtros', 'int', 1),
         ('taxa de aprendizado', 'float', 0.0),
         ('momento', 'float', 0.0),
         ('decaimento de peso', 'float', 0.0)
     ], 'generateOut': checkconvNc, 'ok': False},
    {'name': 'Pooling', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('passo', '2dim', [1, 1]),
         ('filtro', '2dim', [1, 1]),
     ], 'generateOut': checkPool, 'ok': False},
    {'name': 'PoolingAv', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('passo', '2dim', [1, 1]),
         ('filtro', '2dim', [1, 1]),
     ], 'generateOut': checkPool, 'ok': False},
    {'name': 'Padding', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('top', 'int', 1),
         ('bottom', 'int', 1),
         ('left', 'int', 1),
         ('right', 'int', 1),
     ], 'generateOut': checkPadding, 'ok': False},
    {'name': 'BatchNorm', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('epsilon', 'float', 1e-10),
     ], 'generateOut': checkGeneric, 'ok': False},
    {'name': 'SoftMax', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
     ], 'generateOut': checkGeneric, 'ok': False},
    {'name': 'DropOut', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('probabilidade de saida', 'float', 0.5),
     ], 'generateOut': checkGeneric, 'ok': False},
    {'name': 'FullConnect', 'saida': [0, 0, 0],
     'args': [
         ('typeMemory', 'options', ['NO COPY', 'Shared Mem'], [0, 1]),
         ('saida', '1dim', [1]),
         ('função de ativação', 'options', ['SIGMOID', 'TANH', 'RELU'], [0, 2, 4]),
         ('taxa de aprendizado', 'float', 0.0),
         ('momento', 'float', 0.0),
         ('decaimento de peso', 'float', 0.0)
     ], 'generateOut': checkFull, 'ok': False},
]
