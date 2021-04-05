import socket

def splitLen(string,n):
    return  [string[y-n:y] for y in range(n,len(string)+n,n)]    
def ints2bytes(numbers: list,nbyte=4,signed=False):
    b = b''
    for nb in numbers:
        n = int(nb)
        b = b+n.to_bytes(length=nbyte, byteorder='big', signed=signed)
    return b
def bytes2ints(binary_data: bytes,nbytes=4,signed=False):
    byte = splitLen(binary_data,nbytes)
    return [int.from_bytes(bt, byteorder='big', signed=signed) for bt in byte]

# IMPORTAR BIBLIOTECA DA REDE CONVOLUCIONAL
from CNN_GPU.CNN import *
# LER REDE TREINADA
cnn = CNN.load('redeTreinada.cnn')
cnn.compile()

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
            print('espero receber',size)
            # size bytes contendo a imagem
            imagem = con.recv(size)
            print('recebi ',len(imagem))
            if not imagem: break
            imagem = bytes2ints(imagem,1)       
            
            # normalizar imagem
            imagem = cnn.normalizeVectorKnowedSpace(imagem,255,0,1,0)
            
            #Avaliar imagem para descobrir o valor
            cnn.predict(imagem)
            
            # pegar saida
            ans = cnn.getOutputAsIndexMax()
            
            # obtendo saida dos filtros
            '''
            x,y,z,_ = cnn.getSizeData(0,REQUEST_OUTPUT)
            dt = cnn.getData(0,REQUEST_OUTPUT)
            
            sdt = splitLen(dt,x*y)
            dt = []
            # normalizando saidas dos filtros
            for k in range(z):
                sdt[k] = cnn.normalizeVector(sdt[k],255,0)
                dt = dt+sdt[k]          
            
            # montando saida
            dimensao = [x,y,z]
            bsaida = ints2bytes([ans],1)+ints2bytes(dimensao,4)+ints2bytes(dt,1)
            print('enviando resposta de ',len(bsaida),'bytes')
            bsaida = ints2bytes([len(bsaida)],4)+bsaida
            '''
            x,y,dt = cnn.getOutPutAsPPM()
            print(x,y)
            print(len(dt),x*y)
            dimensao = [x,y]
            bsaida = ints2bytes([ans],1)+ints2bytes(dimensao,4)+dt
            
            print('enviando resposta de ',len(bsaida),'bytes')
            bsaida = ints2bytes([len(bsaida)],4)+bsaida
            con.send(bsaida)
            
            print('enviados')
    except Exception as e:
        print( 'Finalizando conexao do cliente', cliente)
        con.close()
        print(e)
    

