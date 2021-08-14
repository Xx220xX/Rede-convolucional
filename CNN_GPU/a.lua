-- Diretorio inicial
home  = [[C:\Gabriela\TESTES_REDE_CONVOLUCIONAL\treino_10classes - escala de cinza]]

-- Nome do treinamento
nome = 'Cifar_10_cinza_'



-- epocas e parametros de treinamento
Numero_epocas = 100
SalvarBackupACada = 50000
Numero_Imagens =  60000
Numero_ImagensTreino = 50000
Numero_ImagensAvaliacao = Numero_Imagens-Numero_ImagensTreino
Numero_Classes = 10
classes = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'}
taxaAprendizado = 0.1;

bytes_remanessentes_imagem = 0
bytes_remanessentes_classes = 0

SalvarSaidasComoPPM =  10 -- somente as 10 primeiras (ainda nao implementado)




-- saidas de arquivos
estatisticasDeTreino = nome..'estatisticasTreino.md'
estatiscasDeAvaliacao = nome..'estatisticasAvaliacao.md'

-- entradas
arquivoContendoImagens = 'imagesCifar10.ubyte'
arquivoContendoRespostas = 'labelsCifar10.ubyte'

Args('work_path',home)
Args("file_image",arquivoContendoImagens)
Args("file_label",arquivoContendoRespostas)
Args("header_image",bytes_remanessentes_imagem)
Args("header_label",bytes_remanessentes_classes)

Args("numero_epocas",Numero_epocas)
Args("numero_imagens",Numero_Imagens)
Args("numero_treino",Numero_ImagensTreino)
Args("numero_fitnes",Numero_ImagensAvaliacao)
Args("numero_classes",Numero_Classes)
Args("sep",32)

local nome_classes_lua
local sep_lua
sep_lua = ' '
if sep ~= nil then sep_lua = sep end
for i,v in pairs(classes) do
	if nome_classes_lua == nil then
		nome_classes_lua = v
	else
		nome_classes_lua = nome_classes_lua ..sep_lua ..v
	end
end
Args("nome_classes",nome_classes_lua)