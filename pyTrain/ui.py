from threading import Thread

from PyQt6 import QtWidgets, uic
import sys, os

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from gab_py_c.gab_cnn import *


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

	def __init__(self):
		super(Ui, self).__init__()
		self.manage = None

		#  UI
		uic.loadUi('uis/main.ui', self)
		self.checkableWindow = [x for x in self.__dict__.keys() if x.startswith('checkWin')]
		for checkableWin in self.checkableWindow:
			self.__dict__[checkableWin].changed.connect(self.checkWindow)
		self.checkWindow()
		self.layerStatus.setVisible(False)
		self.runLua.clicked.connect(lambda x: self.loadLuaFile())

		self.show()

	def checkWindow(self):
		for chw in self.checkableWindow:
			self.__dict__[chw.replace('checkWin', 'window')].setVisible(self.__dict__[chw].isChecked())

	def reportError(self, errorMsg):
		pass

	def loadLuaFile(self, luaFile=None):
		if luaFile is None:
			luaFile = self.ide.toPlainText()
		self.checkWinEditText.setChecked(False)
		Thread(target=self.updateManageData, args=(luaFile,), daemon=True).start()

	def loadProject(self, file, execute=False):
		if not os.path.isfile(file):
			self.reportError('Arquivo não encontrado:' + str(file))
			return False
		txt = open(file, 'r').read()
		self.ide.setPlainText(txt)

	def updateManageData(self, luaFile):
		self.manage = ManageTrain(luaFile, luaisFile=False)
		self.checkWinTrain.setChecked(True)
		entries = ['Imagens para Treino %d' % (int(self.manage.n_images2train),),
				   'Número de epocas %d' % (int(self.manage.n_epics),),
				   'Arquivo de imagem "%s"' % (self.manage.file_images.d.decode('utf-8'),),
				   'Arquivo de etiqueta "%s"' % (self.manage.file_labels.d.decode('utf-8'),),
				   ]

		for en in entries:
			self.listTrain.addItem(en)

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

		self.manage.setEvent(self.manage.UpdateTrain, updateTrain)
		self.manage.trainStart(True)
		self.manage.startLoop(True)

	def startLoadImage(self):
		if self.manage is None:
			self.reportError('Não exite um gerenciador de treino')
			return False


app = QtWidgets.QApplication([sys.argv[0]])
window = Ui()
if len(sys.argv) > 1:
	window.loadProject(sys.argv[1])
app.exec()
