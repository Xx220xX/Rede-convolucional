import time
import urllib.request

import gab_version

lversion = '3.0.0012'
exelink = ('https://github.com/Xx220xX/Rede-convolucional/raw/master/RELEASE/GAB/bin/Gab.exe', 'Gab.exe')
dlllink = ('https://github.com/Xx220xX/Rede-convolucional/raw/master/RELEASE/GAB/bin/gabkernel.dll', 'gabkernel.dll')


class DownloadProgrees:
	def __init__(self):
		self.t0 = time.time()
		self.b = 0

	def __call__(self, count, blockSize, totalSize):
		self.b = self.b + blockSize
		t = time.time() - self.t0
		print("\rBaixando %.2f%%" % (self.b / totalSize * 100,), end='')
		print("%.2f kbps"%(self.b / t / 1024,), end='')


def update():
	print("-" * 40)
	print(f'Baixando {exelink[1]}')
	urllib.request.urlretrieve(*exelink, reporthook=DownloadProgrees())
	print("\n")
	print(f'Baixando {dlllink[1]}')
	urllib.request.urlretrieve(*dlllink, reporthook=DownloadProgrees())
	print("\n")


if gab_version != lversion:
	update()
