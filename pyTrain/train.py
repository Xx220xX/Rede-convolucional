import tkinter as tk

from Activity import *

from treinoui import ui
from gab_py_c import ManageTrain
import sys

print(sys.argv)
exit(0)
root = tk.Tk()
root.geometry('%dx%d' % (root.winfo_screenwidth(), root.winfo_screenheight()))
Activity(root, 'Treino', ui, width=root.winfo_screenwidth(), height=root.winfo_screenheight()).start()
root.mainloop()
