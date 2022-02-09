import os

os.system('windres icone.rc -O coff -o external//lib//icone.o')
print('rc compilado')