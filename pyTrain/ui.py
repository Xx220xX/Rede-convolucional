from threading import Thread

from PyQt6 import QtWidgets, uic
import sys, os, queue

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from gab_py_c.gab_cnn import *


def after(func, ms, *args, **kwargs):
	time.sleep(ms / 1000)
	func(*args, **kwargs)


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

	def __init__(self):
		super(Ui, self).__init__()
		self.manage = None

		# control
		self.runing = False

		self.end = None

		#  UI
		uic.loadUi('uis/main.ui', self)
		# self.queue = queue.Queue()
		self.statusProgress.blockSignals(True)
		# connect(self.execute_funs, Qt.QueuedConnection)

		self.checkableWindow = [x for x in self.__dict__.keys() if x.startswith('checkWin')]
		for checkableWin in self.checkableWindow:
			self.__dict__[checkableWin].changed.connect(self.checkWindow)
		self.checkWindow()
		self.layerStatus.setVisible(False)
		self.runLua.clicked.connect(lambda x: self.loadLuaFile())

		self.show()

	def execute_funs(self):
		while not self.queue.empty():
			(fn, args, kwargs) = self.queue.get()
			fn(*args, **kwargs)

	def call(self, fn, *args, **kwargs):
		"""Schedule a function to be called on the main thread."""
		self.queue.put((fn, args, kwargs))
		self.sig.emit()

	def checkWindow(self):
		for chw in self.checkableWindow:
			self.__dict__[chw.replace('checkWin', 'window')].setVisible(self.__dict__[chw].isChecked())

	def reportError(self, errorMsg):
		print(errorMsg)

	# call back button run editfile
	def loadLuaFile(self):
		if self.runing:
			self.reportError('Existe um processo em segundo plano')
			return False
		luaFile = self.ide.toPlainText()
		if not (self.manage is None):
			self.end = None
			try:
				self.manage.setRun(False)
				del self.manage
				self.manage = None
			except Exception as e:
				print(e)
		if self.checkWinEditText.isChecked():
			self.checkWinEditText.setChecked(False)
		if not self.layerStatus.isVisible():
			self.layerStatus.setVisible(True)
		self.statusProgress.setValue(0)
		self.status.setText("status: Carregando script lua")

		def close(*args):
			self.statusProgress.setValue(100)
			self.startLoadImage()

		self.end = close
		Thread(target=self.updateManageData, args=(luaFile,), daemon=True).start()

	# carregar arquivo lua
	def loadProject(self, file, execute=False):
		if not os.path.isfile(file):
			self.reportError('Arquivo não encontrado:' + str(file))
			return False
		txt = open(file, 'r').read()
		self.ide.setPlainText(txt)

	def updateManageData(self, luaFile):
		self.manage = ManageTrain(luaFile, luaisFile=False)

		@EVENT
		def end(manageTrain):
			try:
				self.end(manageTrain)
			except Exception:
				pass

		self.manage.setEvent(self.manage.OnFinishLoop, end)
		if not self.checkWinTrain.isChecked():
			self.checkWinTrain.setChecked(True)

		entries = ['Imagens para Treino %d' % (int(self.manage.n_images2train),),
				   'Número de epocas %d' % (int(self.manage.n_epics),),
				   'Arquivo de imagem "%s"' % (self.manage.file_images.d.decode('utf-8'),),
				   'Arquivo de etiqueta "%s"' % (self.manage.file_labels.d.decode('utf-8'),),
				   ]
		self.listTrain.clear()
		for en in entries:
			self.listTrain.addItem(en)
		if not (self.end is None):
			self.end()

	def startTrain(self):
		if self.manage is None:
			self.reportError('Não exite um gerenciador de treino')
			return False
		self.manage.setRun(True)

		@EVENT
		def updateTrain(manage: ManageTrain):
			imps = float(manage.et.tr_imagem_atual) / float(manage.et.tr_time)
			imagePercent = float(manage.et.tr_numero_imagens) / float(manage.et.tr_numero_imagens) * 100
			self.train_imps.setText("%.3f imps" % (imps,))
			self.train_epic_progress.setValue(imagePercent)
			time.sleep(0.2)

		def end(*args):
			self.runing = False
			self.end = None

		self.manage.setEvent(self.manage.UpdateTrain, updateTrain)
		self.runing = True

		self.manage.trainStart(True)
		self.manage.startLoop(True)

	def startLoadImage(self):
		if self.manage is None:
			self.reportError('Não exite um gerenciador de treino')
			return False
		if self.manage.n_images <= 0:
			self.reportError('Nenhuma imagem para ser carregada')
			return False
		self.runing = True
		self.layerStatus.setVisible(True)
		self.statusProgress.setValue(0)
		self.status.setText('Carregando imagems')

		@EVENT
		def update(mt: c.c_void_p):
			mt = c.cast(mt, c.POINTER(ManageTrain))[0]
			mt: ManageTrain
			v = 50 / mt.n_images * (float(mt.et.ll_imagem_atual) + float(mt.et.ld_imagem_atual))
			try:
				self.statusProgress.setValue(int(v))
			except Exception:
				pass

			try:
				self.status.setText('Carregando imagens' if v < 50 else 'Carregando etiquetas')
			except Exception:
				pass

		# print(v)

		def finishLoadImage(*args):
			self.runing = False
			self.layerStatus.setVisible(False)
			self.end = None

		self.end = finishLoadImage
		self.manage.setEvent(self.manage.UpdateLoad, update)
		self.runing = True
		self.manage.loadImageStart(True)
		self.manage.startLoop(False)


usePythread()
app = QtWidgets.QApplication([sys.argv[0]])
window = Ui()
if len(sys.argv) > 1:
	window.loadProject(sys.argv[1])
app.exec()
