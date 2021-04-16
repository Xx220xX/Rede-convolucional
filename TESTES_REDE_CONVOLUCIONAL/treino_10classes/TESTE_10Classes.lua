-- Diretorio inicial
home  = 'C:/Users/Henrique/Desktop/CNN/TESTES_REDE_CONVOLUCIONAL/treino_10classes'

-- Nome do treinamento
nome = 'Cifar_10_'



-- epocas e parametros de treinamento
Numero_epocas = 10
SalvarBackupACada = 5
Numero_Imagens =  60000
Numero_ImagensTreino = 50000
Numero_ImagensAvaliacao = Numero_Imagens-Numero_ImagensTreino
Numero_Classes = 10
classes = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'}
bytes_remanessentes_imagem = 0
bytes_remanessentes_classes = 0

SalvarSaidasComoPPM =  10 -- somente as 10 primeiras




-- saidas de arquivos
estatisticasDeTreino = nome..'estatisticasTreino.md'
estatiscasDeAvaliacao = nome..'estatisticasAvaliacao.md'

-- entradas
arquivoContendoImagens = 'imagesCifar10.ubyte'
arquivoContendoRespostas = 'labelsCifar10.ubyte'



--  Arquitetura  rede
Entrada(32,32,3)
-- CarregarRede('rede.cnn')
Convolucao(1,3,8)
Pooling(1, 2);
Convolucao(1,2,4)
Pooling(1, 2);
FullConnect(50,SIGMOID)

FullConnect(10, SIGMOID)

