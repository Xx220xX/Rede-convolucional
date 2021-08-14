import ctypes as c
import tkinter as tk
import tkinter.font as Font
import tkinter.messagebox as messagebox
from Activity import *


global_var = {}


def ui(act: Activity):
	treino = act.Frame(act.frame, width=act.px(39), height=act.py(100))

	grafico = act.Frame(treino, width=treino.px(100), height=treino.py(50))

	act.label("epoca", "Epoca ", frame=treino, x=treino.px(1), width=treino.px(30), y=10)
	act.ProgressBar('pg_epoca', frame=treino, x=treino.px(32), width=treino.px(50), y=10)
	act.label("tepoca", '0 s', frame=treino, x=treino.px(83), width=treino.px(16), y=10)

	act.label("imagem", "Imagem ", frame=treino, x=treino.px(1), width=treino.px(30), y=40)
	act.ProgressBar('pg_imagem', frame=treino, x=treino.px(32), width=treino.px(50), y=40)
	act.label("timagem", '0 s', frame=treino, x=treino.px(83), width=treino.px(16), y=40)

	act.text('MSE ', frame=treino, x=treino.px(1), width=treino.px(30), y=70)
	act.label('mse', '1', frame=treino, x=treino.px(32), width=treino.px(66), y=70)

	act.text('Imagens por segundo ', frame=treino, x=treino.px(1), width=treino.px(30), y=100)
	act.label('image_p_s', '-', frame=treino, x=treino.px(32), width=treino.px(66), y=100)
	act.text('Acertos ', frame=treino, x=treino.px(1), width=treino.px(30), y=130)
	act.label('win_rate', '0%', frame=treino, x=treino.px(32), width=treino.px(66), y=130)

	act.button('init', 'Começar', frame=treino, x=treino.px(10), width=treino.px(35), y=150)
	act.button('stop', 'Cancelar', frame=treino, x=treino.px(55), width=treino.px(35), y=150)

	act.extras['plt0'] = act.putGraphics(grafico, width=grafico.w, height=grafico.h, title='Info train', data=2,
										 legend=['MSE', 'Acerto'],
										 xlabel='epoca', ylim=[-0.5, 1.5])

	# act.setVar('pg_epoca', 0.5)
	# act.setVar('pg_imagem',0.5)


	grafico.actplace(y=treino.px(40))
	treino.actplace(x=act.px(10))
