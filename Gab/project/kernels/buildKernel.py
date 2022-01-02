import os

path = 'camadas'
output = 'kernels.c'
kernelFiles = ['utils.h'] + [b for b in os.listdir(path) if b != 'utils.h']

with open(output, 'w') as fout:
	fout.write('#include "kernels.h"\n')
	for kernel in kernelFiles:
		print('cl:', kernel)
		with open(path + '/' + kernel, 'r') as fin:
			fout.write('//' + kernel + '\n')
			fout.write(fin.read())
			fout.write('\n')
	fout.write('#endif //GAB_KERNELS_OPENCL_H\n')
print()
with open('defaultkernel.h', 'w') as fout:
	fout.write("//\n"
			   "// Created by Henrique on 14/08/2021.\n"
			   "//\n"
			   )
	fout.write('const char __default_kernel__[] = \n')
	for kernel in kernelFiles:
		l = 1
		with open(path + '/' + kernel, 'r') as fin:
			for line in fin:
				line = line.replace('\\', r'\\').replace('"', r'\"')
				txt = '/*%d*/' % (l,) + '\t\t"' + line.strip('\n') + r'\n"' + '\n'
				l = l + 1
				fout.write(txt)

	fout.write(';\n')

import pathlib
import re

pathkernels = str(pathlib.Path(__file__).parent.resolve()).replace('\\', '/') + '/camadas'
with open('kernels.h', 'w') as fout:
	fout.write("//\n"
			   "// Created by Henrique on 01/01/2022.\n"
			   "//\n"
			   )
	fout.write('#ifndef GAB_KERNEL_H\n')
	fout.write('#define GAB_KERNEL_H\n')
	fout.write('#include <gpu/gpu_macros.h>\n')
	fout.write('#define __kernel\n')
	fout.write('#define __global\n')
	for kernel in kernelFiles:
		fout.write('// cl:%s\n' % (kernel,))
		with open(path + '/' + kernel, 'r') as fin:
			txt = fin.read()
			functions = re.findall(r'kV [\w]+\([^{]+', txt)
			fout.write('\n'.join([x + ';' for x in functions]))
			fout.write('\n')
	fout.write('#endif // GAB_KERNEL_H\n')

with open('../include/cwrap_kernels.h', 'w') as fout:
	for kernel in kernelFiles:
		fout.write('//%s\n' % (kernel,))
		with open(path + '/' + kernel, 'r') as fin:
			txt = fin.read()
			functions = re.findall(r'kV [\w]+\([^{]+', txt)
			fdefines = []
			for f in functions:
				fd = {}
				f = f.replace('kV ', '').strip()
				# encontrar primerio parentese
				i = 0
				while i < len(f) and f[i] != '(': i += 1
				if i >= len(f): raise SyntaxError(f'{f}\n{kernel}')
				fd['name'] = f[:i]
				f = f[i + 1:]
				fd['args'] = f.strip(')')
				s = f'#define Knew_{fd["name"]}(x) KRN_new(x, "{fd["name"]}","{fd["args"]}")\n'
				fout.write(s)
			fout.write('\n')
