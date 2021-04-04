import concurrent.futures as CF
def splitLen(string,n,nhandles = 100):
    def _spt(y):
        return string[y-n:y]
    with CF.ThreadPoolExecutor(nhandles) as executor:
        ans = executor.map(_spt,range(n,len(string)+n,n))
    return list(ans)
    #return  [string[y-n:y] for y in range(n,len(string)+n,n)]    
def ints2bytes(numbers: list,nbyte=4,signed=False):
    b = b''
    for nb in numbers:
        n = int(nb)
        b = b+n.to_bytes(length=nbyte, byteorder='big', signed=signed)
    return b
def bytes2ints(binary_data: bytes,nbytes=4,signed=False):
    byte = splitLen(binary_data,nbytes)
    return [int.from_bytes(bt, byteorder='big', signed=signed) for bt in byte]


# IMPORTAR BIBLIOTECA DA REDE NEURAL
from CNN_GPU.CNN import *

# LER REDE TREINADA
cnn = CNN.load('redeTreinada.cnn')
cnn.compile()

im  =  list(range(0,1026))
print(im[1020:1026])
g = cnn.normalizeVectorKnowedSpace(im,1025,0,1,0)

print(g[1020:1026])
