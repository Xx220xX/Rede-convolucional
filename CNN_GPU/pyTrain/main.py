import tkinter as tk

from pyTrain import Activity


def screen(act: Activity):
    act.text("texto")

root = tk.Tk()
root.geometry('%dx%d'%(root.winfo_screenwidth(), root.winfo_screenheight()))
Activity(root, 'Inicializar Valores', screen, width=root.winfo_screenwidth(), height = root.winfo_screenheight()).start()
root.mainloop()