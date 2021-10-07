import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, \
	QPushButton, QVBoxLayout, QFileDialog


# you can copy and run this code

class MainWindow(QMainWindow):
	def __init__(self, parent=None):  # --------------
		super(MainWindow, self).__init__(parent)  # |
		self.setWindowTitle("open file dialog")  # |
		#												   |
		btn = QPushButton("Open File")  # |---- Just initialization
		layout = QVBoxLayout()  # |
		layout.addWidget(btn)  # |
		widget = QWidget()  # |
		widget.setLayout(layout)  # |
		self.setCentralWidget(widget)  # -------------

		btn.clicked.connect(self.open)  # connect clicked to self.open()
		self.show()

	def open(self):
		path = QFileDialog.getOpenFileName(self, 'Open a file', '',
										   'All Files (*.*)')
		if path != ('', ''):
			print("File path : " + path[0])


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	sys.exit(app.exec())
