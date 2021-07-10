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
	// variavel auxiliar para manipular strings
	int erro;
	//  variavel cnn
	Cnn cnn = NULL;
	//variaveis para imagens de entrada e suas respectivas respostas
	double *input = NULL, *target = NULL;
	// semelhante ao target, porem no modo numerico de 0 a NumeroDeClasses
	unsigned char *targeti = NULL;
	// caso cnn não oi instanciada, vai para o fim do programa


	// muda para o diretorio passado no script lua
	if (!SetCurrentDirectory("C:\\Users\\Henrique\\Desktop\\last\\TESTES_REDE_CONVOLUCIONAL\\treino_10classes")) {
		fprintf(stderr, "Falha ao mudar para o diretorio '%s'\n", "");
		erro = 2;
		goto end;
	}
	printf("create cnn\n");
	int nfiltro =640;
	char usehost = 0;

	cnn = createCnnWithWrapperProgram(default_kernel, (Params) {0.1, 0.3, 0}, 32, 32, 3, CL_DEVICE_TYPE_GPU);
	printf("%d\n",CnnAddConvLayer(cnn,usehost, 1, 3, nfiltro));
	printf("%d\n",CnnAddPoolLayer(cnn,usehost, 1, 2));
	printf("%d\n",CnnAddConvLayer(cnn,usehost, 1, 3, nfiltro));
	printf("%d\n",CnnAddFullConnectLayer(cnn,usehost, nfiltro,FTANH));
	printf("%d\n",CnnAddFullConnectLayer(cnn,usehost, nfiltro,FTANH));
	printf("%d\n",CnnAddFullConnectLayer(cnn,usehost, 10,FTANH));
	if(cnn->target->bytes != 10*sizeof(double)){
		cnn->error.error = -82;
		fprintf(stderr, "%d: %s\n", cnn->error.error,"invalid size output");
		goto end;
	}
	if (cnn->error.error) {
		fprintf(stderr, "%d: %s\n", cnn->error.error, cnn->error.msg);
		goto end;
	}
	printCnn(cnn);

	printf("lendo imagens:");
	int nImagens = 100;
	erro = loadSamples(cnn, &input, &target, &targeti, "imagesCifar10.ubyte", "labelsCifar10.ubyte",
	                   10, nImagens, 0, 0);
	printf("%s\n", erro ? " erro\n" : " sucess\n");
	if (erro)goto end;

	double *img, *label;
	uint8_t *labeli;
	for (int i = 0; i < nImagens; i++) {
		img = input + i * (32 * 32 * 3);
		label = target + i * 10;
		labeli = targeti + i;
		CnnCall(cnn, img);
		CnnLearn(cnn, label);
		CnnCalculeError(cnn);
		if (cnn->error.error) {
			fprintf(stderr, "%d: %s\n", cnn->error.error, cnn->error.msg);
			goto end;
		}
		printf("%lf\n",cnn->normaErro);
	}
	printf("for terminado\n");
	clFinish(cnn->queue);
	printf("gpu terminado\n");


	end:
	if (input)free(input);
	if (target)free(target);
	if (targeti)free(targeti);
	if (cnn) {
		if (cnn->error.error)erro = cnn->error.error;
		releaseCnn(&cnn);
	}
	return erro;
}