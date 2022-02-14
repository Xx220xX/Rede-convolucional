import os
import sys

try:
	import urllib.request

	if '-read' in sys.argv:
		os.system('Gab.exe --updatepy')

	print('hello from py')


	class DownloadProgrees:
		def __call__(self, count, blockSize, totalSize):
			print("\rBaixando %.2f%%" % ((count) * blockSize / totalSize * 100,), end='')


	newslink = ('https://raw.githubusercontent.com/Xx220xX/Rede-convolucional/master/RELEASE/lastversion.py', 'lastversion.py')

	print(f'Baixando {newslink[1]}')
	urllib.request.urlretrieve(*newslink, reporthook=download_progress_hook)
	print()

	import lastversion

	lastversion.update()

	os.remove(newslink[1])
	os.remove('gab_version.py')
except Exception as e:
	print(e)
	input()
