#include "locale.h"
#include"src/treino/treinoWithLua.h"

int main(int nargs, char **args) {
	system("chcp 65001");
	printf("##############################\n");
	printf("Gabriela IA\n");
	printf("email: gab.cnn.ia@gmail.com\n");
	printf("Versão %s\n", getVersion());
	printf("##############################\n");
	// incializa a biblioteca random
	LCG_setSeed(time(NULL));
	if (nargs != 2) {
		fprintf(stderr, "Um script lua de configuração é esperado");
		return -1;
	}

	// variavel auxiliar para manipular strings
	char aux[300] = {0};
	int erro;
	//  variavel cnn
	Cnn cnn = NULL;
	// salva endereço para as funçoes lua instancia-la
	globalcnn = &cnn;

	// o segundo argumento é um arquivo lua de configuração
	char *luaFile = args[1];
	ParametrosCnnALL p = {0};



	// carrega script lua, e inicia a rede neural
	erro = loadLuaParameters(luaFile, &p);
	//variaveis para imagens de entrada e suas respectivas respostas
	double *input = NULL, *target = NULL;
	// semelhante ao target, porem no modo numerico de 0 a NumeroDeClasses
	unsigned char *targeti = NULL;
	// caso aconteça alguma falha vai para o final do programa
	if (erro)goto end;
	// caso cnn não oi instanciada, vai para o fim do programa
	if (!cnn) {
		fprintf(stderr, "Nao foi encontrado uma arquitetura de rede");
		goto end;
	}
	if (cnn->error.error) {
		erro = cnn->error.error;
		fprintf(stderr, "%s\n", cnn->error.msg);
		goto end;
	}
	// por questoes de vazamento de memoria, a  variavel global é resetada
	globalcnn = NULL;

	// printa informações sobre as camadas
	printf("\nArquitetura\n");
	for (int i = 0; i < cnn->size; i++) {
		printf("%s\n", cnn->camadas[i]->toString(cnn->camadas[i]));
	}
	// faz a verificação se a arquitetura está correta
	printf("ESTÁ CORRETO? (S/N)");
	int c = toupper(getchar());
	if (c == 'N') {
		erro = -5;
		goto end;
	}


	// muda para o diretorio passado no script lua
	if (!SetCurrentDirectory(p.home)) {
		fprintf(stderr, "Falha ao mudar para o diretorio '%s'\n", p.home);
		erro = 2;
		goto end;
	}
	// cria diretorios uteis
	printf("Criando diretorios\n");


	createDir("js");
	// salva tabela da arquitetura
	FILE *js_arquitetura = fopen("js/camada.js", "w");
	for (int i = 0; i < cnn->size; i++) {
		fprintf(js_arquitetura, "tablePutColum(tabela_arquitetura,%s);\n",
		        cnn->camadas[i]->getCreateParams(cnn->camadas[i]));
	}
	fclose(js_arquitetura);
	// salvando index
	FILE *js_index = fopen("index.html", "w");
	fprintf(js_index, "%s", INDEX_HTML);
	fclose(js_index);
	printf("carregando imagens:");
	// carrega aas imagens e as respostas
	erro = loadSamples(cnn, &input, &target, &targeti, p.arquivoContendoImagens, p.arquivoContendoRespostas,
	                   p.Numero_Classes, p.Numero_Imagens,
	                   p.bytes_remanessentes_imagem, p.bytes_remanessentes_classes);
	// caso ocorra alguma falha, vai para o final do programa
	if (erro)goto end;
	printf(" ok\n");
	// inicia o processo de treinamento
	printf("iniciando treino\n");
	train(cnn, input, target, targeti, p.Numero_epocas, p.SalvarBackupACada, p.Numero_ImagensTreino,
	      p.estatisticasDeTreino);
	printf("salvando rede\n");

	// salva a rede treinada
	snprintf(aux, 300, "%s.cnn", p.nome);
	FILE *fileCnn = fopen(aux, "wb");
	cnnSave(cnn, fileCnn);
	fclose(fileCnn);

//     avalia a rede
	size_t inputSize = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t outputSize = cnn->camadas[cnn->size - 1]->entrada->x * cnn->camadas[cnn->size - 1]->entrada->y *
	                    cnn->camadas[cnn->size - 1]->entrada->z;
	fitness(cnn, input + p.Numero_ImagensTreino * inputSize, targeti + p.Numero_ImagensTreino,
	        p.Numero_Classes, p.names, p.Numero_ImagensAvaliacao, p.SalvarSaidasComoPPM, p.nome,
	        p.estatiscasDeAvaliacao);

	printf("\ntreino terminado\n");
	// final do programa, libera todos os recursos utilizados
	end:
	if (input)free(input);
	if (target)free(target);
	if (targeti)free(targeti);
	if (cnn) {
		releaseCnn(&cnn);
	}
	if (p.names)free(p.names);
	printf("\n");
//    system("pause");

	return erro;
}