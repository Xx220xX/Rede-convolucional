import re
import pytypes
import os


def removeComents(cprogram):
    s = re.sub(r"((\/\*)[?=\*]*[^(\*\/)]*\*\/)", "", cprogram)
    s = re.sub(r"(\/\/[^\r\n]*)", "", s)
    return s


def detectStruct(msg):
    return re.findall(r"(typedef[\s]+struct[\s]*[\w]*[\s]*{[^}]+}[\s]*[\*\w,\s]+);", msg)


def structGetName(stc):
    title = re.findall(r'(struct[\s\w]*)', stc)
    bottom = re.findall(r'(}[\s\w,\*]+)', stc)
    title = title[0].replace('struct', '').replace(' ', '')
    bottom = re.findall(r'([\w]+)', bottom[0])
    nomes = [title] + bottom
    nomes = [_ for _ in nomes if _ != '']
    if title == '':
        return [nomes[0]]
    if len(nomes) > 1:
        return [nomes[1]]
    return nomes


def structGetvars(s):
    s = re.findall(r'(\{[^\{\}]+\})', s)[0][1:-1]
    s = re.sub(r'[\s]+', ' ', s)
    s = re.findall(r'([\w]+[\s]*[,\*\w\s]+)', s)
    vars = []
    for declaracao in s:
        seq = re.split(r'(^[\w]+)', declaracao)
        tipo = seq[1]
        nomes = seq[2].replace(' ', '').split(',')
        for nome in nomes:
            n = re.findall('\*', nome)
            n = len(n)
            desc = (tipo + '*' * n, nome.replace('*', ''))
            vars.append(desc)
    return vars


def cvarToPy(variables: list):
    pvars = []
    for v in variables:
        tipo, nome = v[0], v[1]
        t = tipo.split('*')
        tp = t.pop(0)
        for _ in t:
            tp = pytypes.toPointer(tp)
        tp = pytypes.getPyType(tp)
        pvars.append((nome, tp))
    return pvars


outfile = open('cfunctions.py', 'w')
ctypes_name = 'c'
print(f"import ctypes as {ctypes_name}", file=outfile)


def putClassInFile(file_h, fpy, ctypes_name='c'):
    c_h = removeComents(open(file_h, 'r').read())
    stcs = detectStruct(c_h)

    for structs in stcs:
        nomes = structGetName(structs)
        variables = structGetvars(structs)
        variablespy = cvarToPy(variables)
        print(f"class {nomes[0]}({ctypes_name}.Structure):", file=fpy)
        print("\t_fields_ = [", file=fpy)
        for v in variablespy:
            print(f"\t\t('{v[0]}',{v[1].replace('{ctypes}', ctypes_name)}),", file=fpy)
        print("\t]", file=fpy)
        for n in nomes[1:]:
            print(f"{n} = {nomes[0]}", file=fpy)


camadas = [r'C:\Gabriela\Rede-convolucional\CNN_GPU\src\gpu\Kernel.h'] + \
          [r'C:\Gabriela\Rede-convolucional\CNN_GPU\src\tensor\Tensor.h'] + \
          [r'C:\Gabriela\Rede-convolucional\CNN_GPU\src\camadas\\' + camada for camada in
           os.listdir(r"/src/camadas") if camada.endswith('.h')]

print("""
def TOPOINTER(c_type):
    tp = c.POINTER(c_type)

    def get(self, item):
        return self[0].__getattribute__(item)

    def set(self, key, value):
        self[0].__setattr__(key, value)

    def rep(self):
        return self[0].__repr__()

    tp.__getattribute__ = get
    tp.__setattr__ = set
    tp.__repr__ = rep

    return tp
""",file=outfile)
for file in camadas:
    putClassInFile(file, outfile)
outfile.close()
