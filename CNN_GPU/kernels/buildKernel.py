import os

path = 'camadas'
output = 'gpu.cl'
kernelFiles = ['utils.h'] + [b for b in os.listdir(path) if b != 'utils.h']

with open(output, 'w')as fout:
    for kernel in kernelFiles:
        print('cl:', kernel)
        with open(path + '/' + kernel, 'r') as fin:
            fout.write('//' + kernel + '\n')
            fout.write(fin.read())
            fout.write('\n')
print()
with open('../src/defaultkernel.h', 'w')as fout:
    fout.write('#ifndef KERNELS_H\n')
    fout.write('#define KERNELS_H\n')
    fout.write('const char default_kernel[] = \n')
    l = 1

    with open( output, 'r') as fin:
        for line in fin:
            line = line.replace('\\', r'\\').replace('"', r'\"')
            txt = '/*%d*/' % (l,) + '\t\t"' + line[:-1] + r'\n"' + '\n'
            l = l + 1
            fout.write(txt)

    fout.write(';\n#endif // KERNELS_H\n')
