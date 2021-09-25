from Activity import *
gabVersion = '2.0.07'

def editText(act: Activity):
	pass


def loadImage(act: Activity):
	pass


def trainUi(act: Activity):
	pass


def learnUi(act: Activity):
	pass


def cnnUi(act: Activity):
	pass


def luaConsoleUi(act):
	pass

def main(act:Activity):
	act.frame.config(bg='#ff00f0')
	act.text('Gab cnn '+gabVersion,act.px(25),y=10,width=act.px(50))
	menubar = tk.Menu(act.frame)
	show_all = tk.BooleanVar(act.frame)
	show_done = tk.BooleanVar(act.frame)
	show_not_done = tk.BooleanVar(act.frame)

	view_menu = tk.Menu(menubar)
	view_menu.add_checkbutton(label="Show All", onvalue=1, offvalue=0, variable=show_all)
	view_menu.add_checkbutton(label="Show Done", onvalue=1, offvalue=0, variable=show_done)
	view_menu.add_checkbutton(label="Show Not Done", onvalue=1, offvalue=0, variable=show_not_done)
	menubar.add_cascade(label='View', menu=view_menu)
	act.root.config(menu=menubar)
	show_all.set(True)

App(main,'Gabriela IA')
