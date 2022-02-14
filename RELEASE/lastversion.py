import urllib.request

import gab_version

lversion = '3.0.0012'
exelink = ('https://github.com/Xx220xX/Rede-convolucional/raw/master/RELEASE/GAB/bin/Gab.exe', 'Gab.exe')
dlllink = ('https://github.com/Xx220xX/Rede-convolucional/raw/master/RELEASE/GAB/bin/Gab.exe', 'Gab.exe')


def download_progress_hook(count, blockSize, totalSize):
	print("\rBaixando %.2f%%" % ((count) * blockSize / totalSize * 100,), end='')


def update():
	print(f'Baixando {exelink[1]}')
	urllib.request.urlretrieve(*exelink, reporthook=download_progress_hook)
	print()


if gab_version != lversion:
	update()
