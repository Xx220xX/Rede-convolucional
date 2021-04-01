from threading  import *
import concurrent.futures as cf
import socket
from math import *

def normalize(data,nThreds=10,maxValue=255,useFloor = True):
    mx =  max(data)
    mn = min(data)
    mx = mx-mn
    if mx == 0:
        return [0]*len(data)
    mx = maxValue/mx
    def _norm(x):
        return (x-mn)* mx
    def _normFloor(x):
        return floor((x-mn)* mx)
    f = _norm
    if useFloor:
        f = _normFloor
    with cf.ThreadPoolExecutor(nThreds) as executor:
         results = executor.map(f, data)
    return list(results)
    
def splitLen(string,n):
    return  [string[y-n:y] for y in range(n,len(string)+n,n)]    
def ints2bytes(numbers: list,nbyte=4,signed=False):
    b = b''
    for n in numbers:
        b = b+n.to_bytes(length=nbyte, byteorder='big', signed=signed)
    return b
def bytes2ints(binary_data: bytes,nbytes=4,signed=False):
    byte = splitLen(binary_data,nbytes)
    return [int.from_bytes(bt, byteorder='big', signed=signed) for bt in byte]


def maxPosition(data):
    p = 0
    for i in range(1,len(data)):
        if data[i]>data[p]:
            p = i
    return p
# IMPORTAR BIBLIOTECA DA REDE NEURAL
from CNN_GPU.CNN import *
# LER REDE TREINADA
c = CNN.load('redeTreinada.cnn')
c.compile()

# CRIAR SERVIDOR 
HOST = ''              # Endereco IP do Servidor
PORT = 5000            # Porta que o Servidor esta
tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
orig = (HOST, PORT)
tcp.bind(orig)
tcp.listen(1)

while True:
    con, cliente = tcp.accept()
    print ('Conectado por', cliente)
    try:
        while True:
            # 4 bytes para o tamanho da msg
            size = con.recv(4) 
            size = bytes2ints(size)[0] 

            # size bytes contendo a imagem
            imagem = con.recv(size)
            if not imagem: break
            imagem = bytes2ints(imagem)       

            # normalizar imagem
            imagem = normalize(imagem,maxValue=1,useFloor=False)

            #Avaliar imagem para descobrir o valor
            c.predict(imagem)
            outputVector = c.output
            print(normalize(outputVector,maxValue=100))
            ans = maxPosition(outputVector)
            # obtendo saida dos filtros
            x,y,z,_ = c.getSizeData(0,REQUEST_OUTPUT)
            dt = c.getData(0,REQUEST_OUTPUT)
            
            f = open("filter.txt",'w')
            for k in range(z):
                for i in range(x):
                    for j in range(y):
                        print('%.1f '%(dt[j+i*y+(x*y)*k],),end='',file=f)
                    print('',file=f)
                print('\n',file=f)
            f.close()

            sdt = splitLen(dt,x*y)
            dt = []
            # normalizando saidas dos filtros
            for k in range(z):
                sdt[k] = normalize(sdt[k],maxValue = 255,useFloor=True)
                dt = dt+sdt[k]
            
            
            
            # montando saida
            dimensao = [x,y,z]
            bsaida = ints2bytes([ans],1)+ints2bytes(dimensao,4)+ints2bytes(dt,1)
            print(len(bsaida))
            bsaida = ints2bytes([len(bsaida)],4)+bsaida
            con.send(bsaida)
    except Exception as e:
        print( 'Finalizando conexao do cliente', cliente)
        con.close()
        print(e)
    

