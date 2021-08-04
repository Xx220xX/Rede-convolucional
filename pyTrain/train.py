import tkinter as tk

from Activity import *

from treinoui import ui

root = tk.Tk()
root.geometry('%dx%d'%(root.winfo_screenwidth(), root.winfo_screenheight()))
Activity(root, 'Inicializar Valores', ui, width=root.winfo_screenwidth(), height = root.winfo_screenheight()).start()
root.mainloop()