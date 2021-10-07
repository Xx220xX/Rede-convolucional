from Activity import *

gabVersion = '2.0.07'


def editText(act: Activity):
	frame:tk.Frame
	frame = act.frame_editLua
	act.text('Edit UI',width=frame.px(100),height=25,frame=frame)


def loadImage(act: Activity):
	frame:tk.Frame
	frame = act.frame_cnn
	act.text('Load UI',width=frame.px(100),height=25,frame=frame)


def trainUi(act: Activity):
	frame:tk.Frame
	frame = act.frame_train
	act.text('treino UI',width=frame.px(100),height=25,frame=frame)


def learnUi(act: Activity):
	frame:tk.Frame
	frame = act.frame_fitness
	act.text('fitnes UI',width=frame.px(100),height=25,frame=frame)


def cnnUi(act: Activity):
	frame:tk.Frame
	frame = act.frame_cnn
	act.text('Cnn UI',width=frame.px(100),height=25,frame=frame)



def luaConsoleUi(act: Activity):
	frame:tk.Frame
	frame = act.frame_luaconsole
	act.text('Lua console',width=frame.px(100),height=25,frame=frame)



def main(act: Activity):
	act.frame.config(bg='#ff00f0')
	act.text('Gab cnn ' + gabVersion, act.px(25), y=10, width=act.px(50))

	act.var_bool_editLua = tk.BooleanVar(act.frame)
	act.var_bool_cnn = tk.BooleanVar(act.frame)
	act.var_bool_train = tk.BooleanVar(act.frame)
	act.var_bool_fitness = tk.BooleanVar(act.frame)
	act.var_bool_luaconsole = tk.BooleanVar(act.frame)

	act.frame_editLua = act.Frame(bg='#fffff0', width=act.px(30), height=act.py(100),frameCond = act.var_bool_editLua.get)
	act.frame_cnn = act.Frame(bg='#ffff00', width=act.px(30), height=act.py(100),frameCond = act.var_bool_cnn.get)
	act.frame_train = act.Frame(bg='#ffff0f', width=act.px(30), height=act.py(100),frameCond = act.var_bool_train.get)
	act.frame_fitness = act.Frame(bg='#fff0ff', width=act.px(30), height=act.py(100),frameCond = act.var_bool_fitness.get)
	act.frame_luaconsole = act.Frame(bg='#fff0f0', width=act.px(30), height=act.py(100),frameCond = act.var_bool_luaconsole.get)
	menubar = tk.Menu(act.frame)


	editText(act)
	trainUi(act)
	learnUi(act)
	cnnUi(act)
	luaConsoleUi(act)


	view_menu = tk.Menu(menubar)
	view_menu.add_checkbutton(label="Edit Lua", onvalue=1, offvalue=0, variable=act.var_bool_editLua, command=act.frame_editLua.setShow)
	view_menu.add_checkbutton(label="Cnn", onvalue=1, offvalue=0, variable=act.var_bool_cnn, command=act.frame_cnn.setShow)
	view_menu.add_checkbutton(label="Train", onvalue=1, offvalue=0, variable=act.var_bool_train, command=act.frame_train.setShow)
	view_menu.add_checkbutton(label="Fitness", onvalue=1, offvalue=0, variable=act.var_bool_fitness, command=act.frame_fitness.setShow)
	menubar.add_cascade(label='View', menu=view_menu)
	act.root.config(menu=menubar)

	act.frame_editLua.__getattribute__('actplace' if act.var_bool_editLua.get() else 'place_forget')()
	act.frame_cnn.__getattribute__('actplace' if act.var_bool_cnn.get() else 'place_forget')()
	act.frame_train.__getattribute__('actplace' if act.var_bool_train.get() else 'place_forget')()
	act.frame_fitness.__getattribute__('actplace' if act.var_bool_fitness.get() else 'place_forget')()
	act.frame_luaconsole.__getattribute__('actplace' if act.var_bool_luaconsole.get() else 'place_forget')()

App(main, 'Gabriela IA')
