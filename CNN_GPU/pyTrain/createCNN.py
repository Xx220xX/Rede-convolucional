import ctypes as c
import tkinter as tk
import tkinter.font as Font

from Activity import *

from layers import *


def ui(act: Activity):
    act.text('Entrada', x=act.px(1), y=act.py(0.3), height=act.py(5))
    act.entry('x', 1, int, x=act.px(5), width=act.px(5), height=act.py(5), justify=tk.CENTER)
    act.entry('y', 1, int, x=act.px(10), width=act.px(5), height=act.py(5), justify=tk.CENTER)
    act.entry('z', 1, int, x=act.px(15), width=act.px(5), height=act.py(5), justify=tk.CENTER)
    fArch = act.Frame(act.frame, width=act.px(30), height=act.py(84))
    fEdit = act.Frame(act.frame, width=act.px(68), height=act.py(90))
    fSet = act.Frame(fEdit, width=fEdit.px(50), height=act.py(90))
    fAdd = act.Frame(act.frame, width=fEdit.w, height=act.py(89))

    lista = act.Listbox(x=20, frame=fArch, width=fArch.w - 20, height=fArch.h, font=Font.Font(size=14))
    act.ScroolbarY(lista, frame=fArch, width=20, height=fArch.h)

    params = act.Listbox(frame=fEdit, x=21, y=act.py(6), width=fEdit.px(50), height=fEdit.py(94),
                         font=Font.Font(size=16))
    act.ScroolbarY(params, frame=fEdit, y=act.py(6), width=20, height=fEdit.py(94))

    possibleLayers = act.Listbox(frame=fAdd, x=21, y=act.py(6), width=fAdd.px(50), height=fAdd.py(94),
                                 font=Font.Font(size=16))
    act.ScroolbarY(possibleLayers, frame=fAdd, y=act.py(6), width=20, height=fAdd.py(94))

    act.label('nomeCamada', "NULL", frame=fEdit, x=fEdit.px(1), y=act.py(0.3), height=act.py(5))

    def compile(*args, **kw):
        fileName = act.askSaveAs()
        if fileName == '':return
        with open(fileName,'w') as file:
            act.extras['arch'].compile(file)

    act.button('make', 'Compilar', frame=fAdd, x=fAdd.px(25), y=act.py(0.3), width=fAdd.px(23), height=act.py(5),
               command=compile)

    act.extras['arch'] = Arquitetura()

    def updateArq(*args, **kw):
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

    act.getEntry('x').bind('<KeyRelease>', updateArq)
    act.getEntry('y').bind('<KeyRelease>', updateArq)
    act.getEntry('z').bind('<KeyRelease>', updateArq)
    for layer in LAYERS:
        possibleLayers.insert(tk.END, layer['name'])
    possibleLayers.selection_set(0)
    fArch.actplace(y=act.py(6))

    def putParam(*args, **kw):
        if params.size() <= 0:
            return
        if params.size() == 0:
            selc = 0
        else:
            selc = params.curselection()
            if len(selc) >= 1:
                selc = selc[0]
            else:
                return
        putArgs(act.extras['layer_select']['args'][selc])

    def addLayer(*args, **kw):
        if possibleLayers.size() <= 0:
            return
        if possibleLayers.size() == 0:
            select = 0
        else:
            select = possibleLayers.curselection()
            if len(select) >= 1:
                select = select[0]
            else:
                return
        act.extras['arch'].add(LAYERS[select])

        def updefault(layer):
            LAYERS[select] = copy.deepcopy(layer)

        act.extras['upDefault'] = updefault
        # fAdd.place_forget()
        updateArq()
        # editLayer(act.extras['arch'].camadas[-1], -1)

    def editLayer(layer, index):
        act.setVar('nomeCamada', layer['name'] + f' {layer["saida"]}')
        act.extras['layer_select'] = layer
        params.bind('<<ListboxSelect>>', None)
        params.delete(0, tk.END)
        for arg in layer['args']:
            params.insert(tk.END, arg[0])
        fEdit.actplace(x=act.px(32))
        params.selection_set(0, 0)
        params.bind('<<ListboxSelect>>', putParam)
        act.extras['lastplace_set_or_edit'] = fEdit
        act.extras['index_edit_layer'] = index
        putParam()

    act.button('addLayer', 'Nova camada', frame=fAdd, x=fAdd.px(1), y=act.py(0.3), width=fAdd.px(23), height=act.py(5),
               command=addLayer)

    def confirm(*args, **kw):
        if act.extras['last_arg_frame'] != None:
            act.extras['last_arg_frame'].place_forget()
            act.extras['last_arg_frame'] = None
        updateArq()
        fEdit.place_forget()
        act.extras['lastplace_set_or_edit'] = fAdd
        fAdd.actplace(x=act.px(32))

    def RemoveLayer(*a, **kw):
        if act.extras['last_arg_frame'] != None:
            act.extras['last_arg_frame'].place_forget()
            act.extras['last_arg_frame'] = None
        act.extras['arch'].remove(act.extras['index_edit_layer'])
        updateArq()
        fEdit.place_forget()
        fAdd.actplace(x=act.px(32))
        act.extras['lastplace_set_or_edit'] = fAdd

    act.extras['last_arg_frame'] = None
    act.button('oklayer', 'confirm', frame=fEdit, x=fEdit.px(85), y=act.py(0.3), width=fEdit.px(9), height=act.py(5),
               command=confirm)
    act.button('removeLayer', 'Remove', frame=fEdit, x=fEdit.px(75), y=act.py(0.3), width=fEdit.px(9), height=act.py(5),
               command=RemoveLayer)
    # fEdit.actplace(x=act.px(32))
    fSet.actplace(x=fEdit.px(50) + 20)
    #### continuar
    putArgs = makeOptions(act, fSet)
    fAdd.actplace(x=act.px(32))

    def listaChange(*args, **kw):
        if lista.size() <= 0:
            return
        if lista.size() == 0:
            selc = 0
        else:
            selc = lista.curselection()
            if len(selc) >= 1:
                selc = selc[0]
            else:
                return
        fAdd.place_forget()
        act.extras['upDefault'] = None
        editLayer(act.extras['arch'].camadas[selc], selc)

    lista.bind('<<ListboxSelect>>', listaChange)


def makeOptions(act: Activity, frame: tk.Frame):
    # 2dim
    def update2dim(*args, **kw):
        try:
            x = act.getValue('arg_value_2dim_x')
            y = act.getValue('arg_value_2dim_y')

            act.extras['arg_value'][2][0] = x
            act.extras['arg_value'][2][1] = y
            if act.extras['upDefault'] != None:
                act.extras['upDefault'](act.extras['layer_select'])
        except Exception as e:
            print(e)

    def updatefloat(*args, **kw):
        try:
            floatvalue = act.getValue('arg_value_float')
            act.extras['arg_value'][2] = floatvalue
            if act.extras['upDefault'] != None:
                act.extras['upDefault'](act.extras['layer_select'])
        except Exception:
            return

    def updateint(*args, **kw):
        try:
            intvalue = act.getValue('arg_value_int')
            act.extras['arg_value'][2] = intvalue
            if act.extras['upDefault'] != None:
                act.extras['upDefault'](act.extras['layer_select'])
        except Exception:
            return

    f2dim = act.Frame(frame, width=frame.w, height=frame.py(50))
    act.label('arg_name', 'null', frame=frame, x=frame.px(25), width=frame.px(50), y=frame.py(25) - 30)
    act.entry('arg_value_2dim_x', 1, int, frame=f2dim, y=30, x=f2dim.px(25), width=f2dim.px(50),
              justify=tk.CENTER).bind('<KeyRelease>', update2dim)
    act.entry('arg_value_2dim_y', 1, int, frame=f2dim, y=60, x=f2dim.px(25), width=f2dim.px(50),
              justify=tk.CENTER).bind('<KeyRelease>', update2dim)

    ffloat = act.Frame(frame, width=frame.w, height=frame.py(50))
    act.entry('arg_value_float', 1, float, frame=ffloat, y=30, x=ffloat.px(25), width=ffloat.px(50),
              justify=tk.CENTER).bind('<KeyRelease>', updatefloat)
    fint = act.Frame(frame, width=frame.w, height=frame.py(50))
    act.entry('arg_value_int', 1, int, frame=fint, y=30, x=fint.px(25), width=fint.px(50),
              justify=tk.CENTER).bind('<KeyRelease>', updateint)

    foptions = act.Frame(frame, width=frame.w, height=frame.py(50))

    lista = act.Listbox(x=foptions.px(25), width=foptions.px(50), height=foptions.w, frame=foptions)
    act.ScroolbarY(lista, x=foptions.px(25) - 20, width=20, height=foptions.w, frame=foptions)

    def putArgs(arg):
        if act.extras['last_arg_frame'] != None:
            act.extras['last_arg_frame'].place_forget()
        act.setVar('arg_name', arg[0])
        act.extras['arg_value'] = arg
        if arg[1] == 'float':
            act.setVar('arg_value_float', arg[2])
            ffloat.actplace(y=frame.py(25))
            act.extras['last_arg_frame'] = ffloat
        elif arg[1] == 'int':
            act.setVar('arg_value_int', arg[2])
            fint.actplace(y=frame.py(25))
            act.extras['last_arg_frame'] = fint
        elif arg[1] == '2dim':
            act.setVar('arg_value_2dim_x', arg[2][0])
            act.setVar('arg_value_2dim_y', arg[2][1])
            f2dim.actplace(y=frame.py(25))
            act.extras['last_arg_frame'] = f2dim
        elif arg[1] == 'options':
            lista.delete(0, tk.END)
            for op in arg[3]:
                lista.insert(tk.END, op)
            lista.selection_set(arg[2])
            foptions.actplace(y=frame.py(25))
            act.extras['last_arg_frame'] = foptions

    def listaChange(*args, **kw):
        if lista.size() <= 0:
            return
        if lista.size() == 0:
            selc = 0
        else:
            selc = lista.curselection()
            if len(selc) >= 1:
                selc = selc[0]
            else:
                return
        act.extras['arg_value'][2] = act.extras['arg_value'][4][selc]

    lista.bind('<<ListboxSelect>>', listaChange)

    return putArgs
