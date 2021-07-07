#include"src/treino/treinoWithLua.h"
int main(int nargs, char **args) {
	printf("##############################\n");
	printf("Gabriela IA\n");
	printf("email: gab.cnn.ia@gmail.com\n");
	printf("Versao %s\n",getVersion());
	printf("##############################\n");
	// incializa a biblioteca random
    LCG_setSeed(time(NULL));
    if (nargs != 2) {
        fprintf(stderr, "Um script lua de configuracao e esperado");
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
    if(cnn->error.error){
    	erro = cnn->error.error;
	    fprintf(stderr, "%s\n",cnn->error.msg);
	    goto end;
    }
    // por questoes de vazamento de memoria, a  variavel global é resetada
    globalcnn = NULL;

    // printa informações sobre as camadas
	printf("\nArquitetura\n");
    for (int i = 0; i < cnn->size; i++) {
        printf("%s\n",cnn->camadas[i]->toString(cnn->camadas[i]));
    }
    // faz a verificação se a arquitetura está correta
    printf("ESTA CORRETO? (S/N)");
    int c = toupper(getchar());
    if (c == 'N') {
        erro = -5;
        goto end;
    }

    // flag para calcular o erro durante treinamento
    cnn->flags = CNN_FLAG_CALCULE_ERROR;

    // muda para o diretorio passado no script lua
    if (!SetCurrentDirectory(p.home)) {
        fprintf(stderr, "Falha ao mudar para o diretorio '%s'\n", p.home);
        erro = 2;
        goto end;
    }
	// cria diretorios uteis
	printf("Criando diretorios\n");
	createDir("imgs");
	createDir("redes");
	createDir("js");

    // carrega aas imagens e as respostas
    erro = loadSamples(cnn, &input, &target, &targeti, p.arquivoContendoImagens, p.arquivoContendoRespostas,
                       p.Numero_Classes, p.Numero_Imagens,
                       p.bytes_remanessentes_imagem, p.bytes_remanessentes_classes);
    // caso ocorra alguma falha, vai para o final do programa
    if (erro)goto end;
    // inicia o processo de treinamento
    train(cnn, input, target, targeti, p.Numero_epocas, p.SalvarBackupACada, p.Numero_ImagensTreino,
          p.estatisticasDeTreino);
    // salva a rede treinada
    snprintf(aux, 300, "%s.cnn", p.nome);
    FILE *fileCnn = fopen(aux, "wb");
    cnnSave(cnn, fileCnn);
    fclose(fileCnn);

    // avalia a rede
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
    system("pause");

    return erro;
}