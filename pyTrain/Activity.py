from threading import Thread, Event
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import time


class Activity(Thread):
    def __init__(self, root: tk.Tk, name, screen, x=0, y=0, width=500, height=500):
        self.screen = screen
        self.root = root
        self.frame = tk.Frame(self.root)
        self.x, self.y, self.width, self.height = x, y, width, height
        Thread.__init__(self, target=self.__run__, name=name)
        self.lock = Event()
        self.setDaemon(True)
        self.stoped = False
        self.labels = {}
        self.vars = {}
        self.entrys = {}
        self.tips = {}
        self.buttons = {}
        self.progress = {}
        self.ids = []
        self.name = name
        self.finish = None
        self.w = width
        self.h = height
        self.extras = {}

    def start(self, *args):
        self.args = args
        self.root.title(self.name)
        Thread.start(self)

    def __run__(self):
        self.frame.place(x=self.x, y=self.y, width=self.width, height=self.height)
        # try:
        self.screen(self, *self.args)
        # except Exception as e:
        #     print('erro ao rodar Activity')
        #     print(e)
        #     print('Finalizand todas as telas')
        #     time.sleep(2)
        #     self.root.after(1, self.root.destroy)

    def stop(self):
        def end():
            self.stoped = True
            dest = self.frame.destroy
            self.root.after(1, dest)

        Thread(target=end).start()

    def px(self, porcentagem, xmin=1, w=None):
        if w is None: w = self.w
        x = w * porcentagem / 100
        if x < xmin: return xmin
        return x

    def py(self, porcentagem, ymin=1, h=None):
        if h is None: h = self.h
        y = h * porcentagem / 100
        if y < ymin: return ymin
        return y

    def text(self, text: str, x=0, y=0, width=None, height=None, frame=None, **kwargs):
        if frame is None: frame = self.frame
        label = tk.Label(frame, text=text, **kwargs). \
            place(x=x, y=y, width=width, height=height)
        return label

    def label(self, id, value, gtipo=str, x=0, y=0, width=None, height=None, frame=None, **kwargs):
        if id in self.ids:
            self.labels[id].destroy()
            del self.labels[id], self.vars[id]
        if frame is None: frame = self.frame
        self.ids.append(id)
        self.tips[id] = gtipo
        self.vars[id] = tk.Variable(frame, value=value)
        self.labels[id] = tk.Label(frame, textvariable=self.vars[id], **kwargs)
        self.labels[id].place(x=x, y=y, width=width, height=height)
        return self.labels[id]

    def entry(self, id, value, gtipo=str, x=0, y=0, width=None, height=None, frame=None, **kwargs):
        if id in self.ids:
            self.entrys[id].destroy()
            del self.entrys[id], self.vars[id]
        if frame is None: frame = self.frame
        self.ids.append(id)
        self.tips[id] = gtipo
        self.vars[id] = tk.Variable(frame, value=value)
        self.entrys[id] = tk.Entry(frame, textvariable=self.vars[id], **kwargs)
        self.entrys[id].place(x=x, y=y, width=width, height=height)
        return self.entrys[id]

    def button(self, id, text, x=0, y=0, width=None, height=None, frame=None, **kwargs):
        if id in self.ids:
            self.buttons[id].destroy()
            del self.buttons[id]
        if frame is None: frame = self.frame
        self.ids.append(id)
        self.buttons[id] = tk.Button(frame, text=text, **kwargs)
        self.buttons[id].place(x=x, y=y, width=width, height=height)
        return self.buttons[id]

    def ProgressBar(self, id, x=0, y=0, width=None, height=None, frame=None, max=1.0, **kwargs):
        if frame is None: frame = self.frame
        self.ids.append(id)
        self.vars[id] = tk.DoubleVar(frame, value=0)
        self.progress[id] = ttk.Progressbar(frame, variable=self.vars[id], maximum=max, **kwargs).place(x=x, y=y,
                                                                                                        width=width,
                                                                                                        height=height)
        return self.progress[id]

    def ScroolbarY(self, listbox: tk.Listbox, x=0, y=0, width=None, height=None, frame=None):
        if frame is None: frame = self.frame
        scroll = tk.Scrollbar(frame)
        listbox.config(yscrollcommand=scroll.set)
        scroll.config(command=listbox.yview)
        scroll.place(x=x, y=y, width=width, height=height)
        return scroll

    def ScroolbarX(self, listbox: tk.Listbox, x=0, y=0, width=None, height=None, frame=None):
        if frame is None: frame = self.frame
        scroll = tk.Scrollbar(frame)
        listbox.config(xscrollcommand=scroll.set)
        scroll.config(command=listbox.xview)
        scroll.place(x=x, y=y, width=width, height=height)
        return scroll

    def Listbox(self, x=0, y=0, width=None, height=None, frame=None, Lheight=None, *args, **kw):
        if frame is None: frame = self.frame
        listb = tk.Listbox(frame, *args, height=Lheight, **kw)
        listb.place(x=x, y=y, width=width, height=height)
        return listb

    def Frame(self, frameParente=None, width=None, height=None, *args, **kwargs):
        if frameParente is None: frameParente = self.frame
        if width is None: width = self.w
        if height is None: height = self.h
        frame = tk.Frame(frameParente, *args, **kwargs)
        frame.__dict__['w'] = width
        frame.__dict__['h'] = height

        def put(*args, width=None, height=None, **kwargs):
            if width is None: width = frame.w
            if height is None: height = frame.h
            frame.place(*args, width=width, height=height, **kwargs)

        def px(*args, **kw):
            return Activity.px(frame, *args, **kw)

        def py(*args, **kw):
            return Activity.py(frame, *args, **kw)

        frame.__dict__['actplace'] = put
        frame.__dict__['px'] = px
        frame.__dict__['py'] = py
        return frame
    def askSaveAs(self, types=None):
        if types is None:
            types = [('lua', '*.lua'),('all files','*.*')]
        return filedialog.asksaveasfilename(title='arquitetura',defaultextension=types,filetypes=types)
    def getButton(self, id) -> tk.Button:
        return self.buttons[id]

    def getValue(self, id):
        self.vars[id].get()
        return self.tips[id](self.vars[id].get())

    def getEntry(self, id) -> tk.Entry:
        return self.entrys[id]

    def type(self, v):
        if isinstance(v, int):
            return int
        if isinstance(v, float):
            return float
        if isinstance(v, str):
            return str
        return eval

    def getVar(self, id):
        return self.vars[id]

    def setVar(self, id, value):
        self.frame.after(1, self.vars[id].set, value)

    def setVarAppend(self, id, value):
        self.vars[id].set(self.vars[id].get() + value)

    def finishprogram(self, tm):
        def end():
            self.root.after(tm, self.root.destroy)

        self.finish = end

    @staticmethod
    def timeSTR(t):
        t = int(t)
        s = f' {t % 60} seg'
        t = t // 60
        if t > 0:
            s = f'{t % 60} min ' + s
            t = t // 60
        if t > 0:
            s = f'{t % 24} hr ' + s
            t = t // 24
        if t > 0:
            s = f'{t} dia ' + s
        return s


class Func:
    def __init__(self, f, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.f = f

    def __call__(self, *args, **kwargs):
        self.f(*self.args, **self.kwargs)