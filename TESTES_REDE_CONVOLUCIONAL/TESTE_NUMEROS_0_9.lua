-- Nome do treinamento
nome = 'numeros0_9'

-- Pasta onde sera carregado e salvado todos arquivos
local_de_trabalho = ''


--  Arquitetura para treinar rede

-- CarregarRede('rede.cnn')

entrada(28,28,1)
Convolucional(1,3,8)
Pooling(1, 2);
Convolucional(1,3,8)
FullConnect(50,SIGMOID)
FullConnect(10, FSIGMOID);


-- epocas e parametros de treinamento
Numero_epocas = 100
SalvarBackupACada = 5  
Numero_Imagens =  60000
Numero_ImagensTreino = 50000
Numero_ImagensAvaliacao = Numero_Imagens-Numero_ImagensTreino

SalvarSaidasComoPPM =  10 -- somente as 10 primeiras 

-- saidas de arquivos
estatisticasDeTreino = 'estatisticasTreino.md'
estatiscasDeAvaliacao = 'estatisticasAvaliacao.md'


-- entradas
arquivoContendoImagens = 'input/train-images.idx3-ubyte'
arquivoContendoRespostas = 'input/train-labels.idx1-ubyte'



