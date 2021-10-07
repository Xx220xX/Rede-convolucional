import time
from threading import Thread

from PyQt6 import QtWidgets, uic
import sys, os, queue

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from gab_py_c.gab_cnn import *


def after(func, ms, *args, **kwargs):
	time.sleep(ms / 1000)
	func(*args, **kwargs)


# class MyListWidget(QListWidget):
# 	def clicked(self, item):
# 		QMessageBox.information(self, "ListWidget", "ListWidget: " + item.text())
back = None


def getInt(x: c.c_uint):
	try:
		return int(x.value)
	except:
		return int(x)


@EVENT
def updateLoad(t):
	try:
		t = c.cast(t, TOPOINTER(ManageTrain))
		t: ManageTrain
		try:
			atual = int(t.et.ll_imagem_atual.value) + int(t.et.ld_imagem_atual.value)
		except Exception as e:
			atual = c.c_int(t.et.ll_imagem_atual).value + c.c_int(t.et.ld_imagem_atual).value
		atual = 50 * atual / t.n_images
		if atual > 50:
			back.ui.runOnUiThread(back.ui.status.setText, 'Carregando Labels')
		back.ui.runOnUiThread(back.ui.statusProgress.setValue, int(atual))
	except Exception as e:
		print(e)


@EVENT
def endLoad(t):
	try:
		back.manage.setEvent(back.manage.OnFinishLoop, 0)
		back.ui.runOnUiThread(back.ui.layerStatus.setVisible, False)
		back.ui.runOnUiThread(back.ui.label_imagens_carregadas.setText, f'Imagens carregas {back.manage.et.ll_imagem_atual + 1}')
		back.ui.runOnUiThread(back.ui.checkWinTrain.setChecked, True)
		back.runing = False
	except:
		pass


@EVENT
def updateTrain(t):
	try:
		t = c.cast(t, TOPOINTER(ManageTrain))
		t: ManageTrain

		nepoca = float(t.et.tr_numero_epocas)
		epoca = float(t.et.tr_epoca_atual)
		nimg = float(t.et.tr_numero_imagens)
		imga = float(t.et.tr_imagem_atual)
		tmp = float(t.et.tr_time)
		acerto = float(t.et.tr_acerto_medio)
		mse = float(t.et.tr_erro_medio)
		pcT = int((imga + nimg * epoca + 1) / (nimg * nepoca) * 100)
		pcE = int(((imga + 1) / nimg * 100))
		imps = float(t.et.tr_imps)
		back.ui.runOnUiThread(back.ui.train_epic_progress.setValue, pcE)
		back.ui.runOnUiThread(back.ui.train_train_progress.setValue, pcT)

		back.ui.runOnUiThread(back.ui.train_imps.setText, 'Imagens por segundo %.2f' % (imps,))
		back.ui.runOnUiThread(back.ui.train_infomse.setText, 'Mse %.2f' % (mse,))
		back.ui.runOnUiThread(back.ui.train_infowin.setText, 'Taxa de acerto %.2f' % (acerto,))
	except Exception as e:
		print('Update Train:', e)


@EVENT
def endTrain(t):
	try:
		back.manage.setEvent(back.manage.OnFinishLoop, 0)
		back.ui.runOnUiThread(back.ui.layerStatus.setVisible, False)
		back.ui.runOnUiThread(back.ui.label_imagens_carregadas.setText, f'Imagens carregas {back.manage.et.ll_imagem_atual + 1}')
		back.ui.runOnUiThread(back.ui.checkWinFitness.setChecked, True)
		back.ui.runOnUiThread(back.ui.buttonStartTrain.setVisible, True)
		back.ui.runOnUiThread(back.ui.buttonEndTrain.setVisible, False)
		back.runing = False
	except:
		pass


class BackEnd:
	manage: ManageTrain

	def __init__(self, ui):
		self.ui = ui
		self.runing = False
		self.manage = None
		global back
		back = self

	def loadFile2Ide(self):
		if self.runing:
			self.ui.reportError('Existe um processo em segundo plano')
			return
		self.runing = True
		self.ui.status.setText('Carregando arquivo')
		file = self.ui.var_luafile

		def f():
			try:
				with open(file, 'r') as fl:
					luaFile = fl.read()
			except Exception as e:
				self.ui.runOnUiThread(self.ui.reportError, str(e))
				return
			self.ui.runOnUiThread(self.ui.ide.setPlainText, luaFile)
			self.ui.runOnUiThread(self.ui.checkWinEditText.setChecked, True)
			self.runing = False

		Thread(name='Load File', target=f, daemon=True).start()

	def exeLuaIde(self):
		if self.runing:
			self.ui.reportError('Existe um processo em Segundo plano')
			return
		self.runing = True
		# https://stackoverflow.com/questions/24035660/how-to-read-from-qtextedit-in-python
		lua = self.ui.ide.toPlainText()
		self.ui.layerStatus.setVisible(True)
		self.ui.status.setText('Executando Script')
		self.ui.statusProgress.setValue(0)
		self.ui.checkWinEditText.setChecked(False)

		def f():
			if self.manage is None:
				self.manage = ManageTrain(lua, luaisFile=False)
				self.ui.runOnUiThread(self.ui.buttonReadImage.setVisible, True)
				self.ui.runOnUiThread(self.ui.listTrain.addItem, f'Epocas {self.manage.n_epics}')
				self.ui.runOnUiThread(self.ui.listTrain.addItem, f'Imagens para treino {self.manage.n_images2train}')
			else:
				self.manage.cnn.luaExecute(lua)

			self.ui.runOnUiThread(self.ui.statusProgress.setValue, 100)
			self.ui.runOnUiThread(self.ui.status.setText, 'Script Executado')
			time.sleep(0.1)
			self.ui.runOnUiThread(self.ui.layerStatus.setVisible, False)
			self.ui.runOnUiThread(self.showCnn)
			self.runing = False

		Thread(name='Executando script', target=f, daemon=True).start()

	def loadImage(self):
		if self.runing:
			self.ui.reportError('Existe um processo em segundo plano')
			return
		self.runing = True
		self.ui.buttonReadImage.setVisible(False)
		self.ui.layerStatus.setVisible(True)
		self.ui.status.setText('Carregando Imagens')
		self.ui.statusProgress.setValue(0)
		self.ui.checkWinEditText.setChecked(False)

		self.manage.setEvent(self.manage.UpdateLoad, updateLoad)
		self.manage.setEvent(self.manage.OnFinishLoop, endLoad)
		self.manage.setRun(True)
		self.manage.loadImageStart(True)
		self.manage.startLoop(True)

	def showCnn(self):
		self.ui.checkWinCNN.setChecked(True)
		listCnn = self.ui.listCnn
		listCnn: QListWidget
		listCnn.clear()
		for i in range(self.manage.cnn.size):
			listCnn.addItem(self.manage.cnn.camadas[i].name())

	def initTrain(self):
		if self.runing:
			self.ui.reportError('Existe um processo em execuçao')
			return
		self.runing = True
		if self.manage is None:
			self.runing = False
			self.ui.reportError('Execute um projeto primeiro')
			return
		if self.manage.et.ll_imagem_atual + 1 != self.manage.n_images:
			self.runing = False
			self.ui.reportError('Carregue as imagens primeiro')
			return
		if self.manage.cnn.size == 0:
			self.runing = False
			self.ui.reportError('Nenhuma camada encontrada na Cnn')
			return

		self.manage.setEvent(self.manage.UpdateTrain, updateTrain)
		self.manage.setEvent(self.manage.OnFinishLoop, endTrain)
		self.ui.buttonStartTrain.setVisible(False)
		self.ui.buttonEndTrain.setVisible(True)
		self.manage.trainStart(True)
		self.manage.startLoop(True)

	def endTrain(self):
		if not self.runing: return
		self.manage.setRun(False)
		self.ui.buttonEndTrain.setVisible(False)


class Ui(QtWidgets.QMainWindow):
	actionTrain: QAction
	ide: QPlainTextEdit
	manage: ManageTrain
	runLua: QPushButton
	listTrain: QListWidget
	windowTrain: QDockWidget
	checkWinTrain: QAction
	train_imps: QLabel
	train_time: QLabel
	train_epic_progress: QProgressBar
	layerStatus: QWidget
	status: QLabel
	statusProgress: QProgressBar
	actionOpenProject: QAction
	signal = pyqtSignal(object)

	def __init__(self):
		super(Ui, self).__init__()
		self.back = BackEnd(self)
		self.var_luafile = None  # 'D:/Henrique/treino_ia/treino_numero_0_9/config_09.lua'
		#  UI
		uic.loadUi('uis/main.ui', self)
		self.signal.connect(self.__execute_func__)
		self.checkableWindow = [x for x in self.__dict__.keys() if x.startswith('checkWin')]
		for checkableWin in self.checkableWindow:
			self.__dict__[checkableWin].changed.connect(self.checkWindow)
		self.checkWindow()
		self.layerStatus.setVisible(False)
		self.buttonReadImage.setVisible(False)

		self.buttonReadImage.clicked.connect(self.back.loadImage)
		self.buttonStartTrain.clicked.connect(self.back.initTrain)
		self.buttonEndTrain.clicked.connect(self.back.endTrain)

		self.runLua.clicked.connect(self.back.exeLuaIde)
		self.actionOpenProject.triggered.connect(self.getfiles)
		self.buttonEndTrain.setVisible(False)

		self.show()

	def runOnUiThread(self, func, *args, **kw):
		self.signal.emit((func, args, kw))

	def __execute_func__(self, info):
		func = info[0]
		args = info[1]
		kw = info[2]
		try:
			func(*args, **kw)
		except Exception as e:
			print(e)

	def checkWindow(self):
		for chw in self.checkableWindow:
			self.__dict__[chw.replace('checkWin', 'window')].setVisible(self.__dict__[chw].isChecked())

	def reportError(self, errorMsg):
		QMessageBox.critical(self, "Error", errorMsg)
		print(errorMsg)

	def getfiles(self):
		# https://newbedev.com/python-open-file-dialog-pyqt6-code-example
		if self.back.runing:
			self.reportError('Existe um processo em segundo plano')
			return
		path = QFileDialog.getOpenFileName(self, 'Open a file', '', 'Lua File (*.lua)')
		if path != ('', ''):
			self.var_luafile = path[0]
			self.back.loadFile2Ide()
		else:
			self.reportError('Falha ao carregar arquivo')
			print("File path : " + path[0])


usePythread()
app = QtWidgets.QApplication([sys.argv[0]])
window = Ui()
if len(sys.argv) > 1:
	window.var_luafile = sys.argv[1]
	window.back.loadFile2Ide()

app.exec()
