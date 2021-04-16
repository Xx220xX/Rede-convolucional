-- Diretorio inicial
home  = 'C:/Users/Henrique/Desktop/CNN/TESTES_REDE_CONVOLUCIONAL/treino_numero_0_9'

-- Nome do treinamento
nome = 'numeros0_9'



-- epocas e parametros de treinamento
Numero_epocas = 2000
SalvarBackupACada = 50000
Numero_Imagens =  60
Numero_ImagensTreino = 50
Numero_ImagensAvaliacao = Numero_Imagens-Numero_ImagensTreino
Numero_Classes = 10
classes = {'zero','um','dois','tres','quatro','cinco','seis','sete','oito','nove'}
bytes_remanessentes_imagem = 16
bytes_remanessentes_classes = 8

SalvarSaidasComoPPM =  100 -- somente as 10 primeiras




-- saidas de arquivos
estatisticasDeTreino = nome..'estatisticasTreino.md'
estatiscasDeAvaliacao = nome..'estatisticasAvaliacao.md'

-- entradas
arquivoContendoImagens = 'train-images.idx3-ubyte'
arquivoContendoRespostas = 'train-labels.idx1-ubyte'



--  Arquitetura  rede
Entrada(28,28,1)
-- CarregarRede('rede.cnn')
Convolucao(1,3,8)
Pooling(1, 2);
Convolucao(1,3,8)
FullConnect(50,SIGMOID)
FullConnect(10, SIGMOID)

