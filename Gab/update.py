import os
import sys
import urllib.request

if not ('-read' in sys.argv):
	os.system('"Gab.exe --updatepy"')


def download_progress_hook(count, blockSize, totalSize):
	print("\rBaixando %.2f%%" % ((count) * blockSize / totalSize * 100,), end='')


newslink = ('https://raw.githubusercontent.com/Xx220xX/Rede-convolucional/master/RELEASE/lastversion.py', 'lastversion.py')

print(f'Baixando {newslink[1]}')
urllib.request.urlretrieve(*newslink, reporthook=download_progress_hook)
print()

import lastversion

lastversion.update()
