from threading import Thread
import tkinter as tk


class Form(Thread):
	root: tk.Tk

	def __init__(self, root, name, formFunc):
		Thread.__init__(self, target=self.makeUi, daemon=True)
		self.func = formFunc
		self.args, self.kw = None, None
		self.root = root
		self.root.title(name)

		self._frame = [root]
		self.frame = root

		self.idText = {}

	def newFrame(self, indexFrameParent=0, *args, **kw) -> tk.Frame:
		self.frame = tk.Frame(self.frame, *args, **kw)
		self._frame.append(self.frame)
		return self.frame

	def Frame(self, index=-1) -> tk.Frame:
		frame = self._frame[index]
		return frame

	def makeUi(self, *args, **kw):
		if self.func:
			self.func(self, *self.args, **self.kw)

	def start(self, *args, **kw):
		self.args = args
		self.kw = kw
		Thread.start(self)

	def Text(self, id, parent=-1, *args, **kw) -> tk.Text:
		if id in self.idText:
			raise Exception(f'já existe uma instância com id {id} ')
		self.idText[id] = tk.Text(self.frame[parent],*args, *kw)
		return self.idText[id]


def App(mainActivityUi, name=''):
	root = tk.Tk()
	root.geometry('%dx%d' % (root.winfo_screenwidth(), root.winfo_screenheight()))
	root.state('zoomed')
	Form(root, name, mainActivityUi).start()
	root.mainloop()
