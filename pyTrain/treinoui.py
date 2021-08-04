import ctypes as c
import tkinter as tk
import tkinter.font as Font
import tkinter.messagebox  as messagebox
from Activity import *
from PIL import  Image,ImageTk
import ctypes as c

global_var = {}
@c.CFUNCTYPE(None)
def updateTrain():
	pass


def ui(act: Activity):
	treino = act.Frame(act.frame, width=act.px(39), height=act.py(100))
	fitnes = act.Frame(act.frame, width=act.px(39), height=act.py(100))
	grafico = act.Frame(treino, width=treino.px(100), height=treino.py(50))

	act.label("epoca","Epoca ",frame=treino,x=treino.px(1),width=treino.px(30),y=10)
	act.ProgressBar('pg_epoca',frame=treino,x=treino.px(32),width=treino.px(50),y=10)
	act.label("tepoca",'0 s',frame=treino,x=treino.px(83),width=treino.px(16),y=10)

	act.label("imagem","Imagem ",frame=treino,x=treino.px(1),width=treino.px(30),y=40)
	act.ProgressBar('pg_imagem',frame=treino,x=treino.px(32),width=treino.px(50),y=40)
	act.label("timagem",'0 s',frame=treino,x=treino.px(83),width=treino.px(16),y=40)

	act.label("imagem_ft","Imagem ",frame=fitnes,x=fitnes.px(1),width=fitnes.px(30),y=40)
	act.ProgressBar('pg_imagem_ft',frame=fitnes,x=fitnes.px(32),width=fitnes.px(50),y=40)
	act.label("timagem_ft",'0 s',frame=fitnes,x=fitnes.px(83),width=fitnes.px(16),y=40)

	act.text('MSE ',frame=treino,x=treino.px(1),width=treino.px(30),y=70)
	act.label('mse','1',frame=treino,x=treino.px(32),width=treino.px(66),y=70)

	act.text('MSE ',frame=fitnes,x=fitnes.px(1),width=fitnes.px(30),y=70)
	act.label('mse_ft','1',frame=fitnes,x=fitnes.px(32),width=fitnes.px(66),y=70)


	act.text('Acertos ',frame=treino,x=treino.px(1),width=treino.px(30),y=100)
	act.label('rate','0%',frame=treino,x=treino.px(32),width=treino.px(66),y=100)

	act.text('Acertos ',frame=fitnes,x=fitnes.px(1),width=fitnes.px(30),y=100)
	act.label('rate_ft','0%',frame=fitnes,x=fitnes.px(32),width=fitnes.px(66),y=100)


	act.button('stop','Cancelar',frame=treino,x=treino.px(25),width=treino.px(50),y=130)
	act.button('stop_ft','Cancelar',frame=fitnes,x=fitnes.px(25),width=fitnes.px(50),y=130)

	act.putGraphics(grafico,width=grafico.w,height=grafico.h,title='Info train',data=2,legend=['MSE','Acerto'],xlabel = 'epoca')

	act.setVar('pg_epoca',0.5)
	# act.setVar('pg_imagem',0.5)


	grafico.actplace(y=treino.px(40))
	treino.actplace(x=act.px(10))
	fitnes.actplace(x=act.px(51))
