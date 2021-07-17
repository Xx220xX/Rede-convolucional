import tkinter as tk

from pyTrain.Activity import Activity
from pyTrain.createCNN import ui

root = tk.Tk()
root.geometry('%dx%d'%(root.winfo_screenwidth(), root.winfo_screenheight()))
Activity(root, 'Criar Arquitetura', ui,width=root.winfo_screenwidth(),height = root.winfo_screenheight()).start()
root.mainloop()