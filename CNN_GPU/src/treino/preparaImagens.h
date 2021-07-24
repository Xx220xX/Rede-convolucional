//
// Created by Henrique on 16/07/2021.
//


void showDir() {
	DIR *dir;
	struct dirent *ent;
	char currentDir[250] = {0};
	GetCurrentDirectory(250, currentDir);
	printf("Current dir%s\n", currentDir);
	/*if ((dir = opendir(currentDir)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	getchar();
	 */
}

/**
 *  Carrega as imagens do arquivo
 * @param cnn  instancia valida de uma cnn (para utilizar a gpu)
 * @param images  ponteiro para vetor double onde as imagens serão salvas
 * @param remainImage  bits de cabeçalho
 * @param numberOfSamples  numero de imagens a ser lida
 * @param imageFile  string contendo o nome do arquivo que contem as imagens
 * @return  0 caso sucesso. -1 caso não seja possivel abrir o arquivo. -2 caso o numero de imagens
 * lidas seja diferente do numero de imagens especificado
 */
int loadImage(Cnn cnn, double **images, size_t remainImage, size_t numberOfSamples, char *imageFile) {
	// obtem o tamanho de cada imagem
	size_t pixelsByImage = cnn->camadas[0]->entrada->x * cnn->camadas[0]->entrada->y * cnn->camadas[0]->entrada->z;
	size_t samples;
	FILE *fimage = fopen(imageFile, "rb");
	if (!fimage) {
		fprintf(stderr, "Imagens nao foram encontradas em %s\n", imageFile);
		*images = NULL;
		return -1;
	}
	*images = (double *) calloc(sizeof(double), pixelsByImage * numberOfSamples);
	fread(*images, 1, remainImage, fimage);// bytes remanessentes de cabeçalho
	normalizeImage(cnn, *images, numberOfSamples * pixelsByImage, fimage, &samples);
	fclose(fimage);
	if (cnn->error.error || numberOfSamples * pixelsByImage != samples) {
		fprintf(stderr, "As imagens nao foram lidas corretamente\n");
		free_mem(*images);
		*images = NULL;
		return cnn->error.error;
	}
	return 0;

}

/**
 *  Carrega as respostas do arquivo
 * @param cnn  instancia valida de uma cnn (para utilizar a gpu)
 * @param labels  ponteiro para vetor double onde as respostas serão salvas
 * @param remainLabel  bits de cabeçalho
 * @param numberOfSamples  numero de respostas a ser lida
 * @param numeroSaidas  numero de classes total
 * @param labelFile  string contendo o nome do arquivo que contem as respostas
 * @return  0 caso sucesso. -1 caso não seja possivel abrir o arquivo. -2 caso o numero de respostas
 * lidas seja diferente do numero de respostas especificado
 */
int loadLabel(Cnn cnn, double **labels, unsigned char **labelsI, size_t remainLabel, size_t numberOfSamples,
              size_t numeroSaidas, char *labelFile) {
	FILE *flabel = fopen(labelFile, "rb");
	if (!flabel) {
		fprintf(stderr, "Labels nao foram encontradas em %s\n", labelFile);
		*labels = NULL;
		*labelsI = NULL;
		return -1;
	}
	*labels = (double *) calloc(sizeof(double), numeroSaidas * numberOfSamples);
	*labelsI = (unsigned char *) calloc(sizeof(unsigned char), numberOfSamples);
	fread(*labels, 1, remainLabel, flabel);

	size_t lidos = 0;
	loadTargetData(cnn, *labels, *labelsI, numeroSaidas, numberOfSamples, flabel, &lidos);
	fclose(flabel);
	if (cnn->error.error || numberOfSamples != lidos) {
		fprintf(stderr, "Esperado %lld, lidos %lld\n", numberOfSamples, lidos);
		*labels = NULL;
		*labelsI = NULL;
		return cnn->error.error;
	}
	return 0;
}

/**
 * Carrega as imagens e as respostas
 * @param cnn   instancia valida de uma cnn (para utilizar a gpu)
 * @param images ponteiro para vetor double onde as imagens serão salvas
 * @param labels ponteiro para vetor double onde as respostas serão salvas
 * @param labelsI ponteiro para vetor char onde as respostas serão salvas
 * @param imageFile string contendo o nome do arquivo que contem as imagens
 * @param labelFile string contendo o nome do arquivo que contem as respostas
 * @param numberOfLabels numero de classes possiveis
 * @param numberOfSamples numero de exemplos a ser carregado
 * @param remainImage  bytes de cabeçalho para imagens
 * @param remainLabel bytes de cabeçalho para respostas
 * @return caso nao contenha erro retorna 0,caso falhe na leitura das imagens retorna -8&erro, caso falhe na leitura das respostas retorna -16&erro
 */
int loadSamples(Cnn cnn, double **images, double **labels, unsigned char **labelsI, char *imageFile, char *labelFile,
                size_t numberOfLabels, size_t numberOfSamples, size_t remainImage, size_t remainLabel) {

	int erro = 0;
	// le imagens
	if ((erro = loadImage(cnn, images, remainImage, numberOfSamples, imageFile)))return 8 & erro;
	// le respostas
	if ((erro = loadLabel(cnn, labels, labelsI, remainLabel, numberOfSamples, numberOfLabels, labelFile)))
		return 16 & erro;
	return 0;
}

/**
 * Salva imagem no formato ppm. Seu nome é sua resposta e contém tmb o vetor correspondente
 * @param images vetor de imagens
 * @param labels vetor de respostas em modo vetor
 * @param labelsI vetor de respostas em modo inteiro
 * @param n numero de imagens a ser salvas
 * @param x dimensao x da imagem (largura)
 * @param y dimensao y da imagem (altura)
 * @param z dimensao z da imagem
 * @param m numero de classes
 */
void printTestImages(double *images, double *labels, unsigned char *labelsI, int n, int x, int y, int z, int m) {
	char buff[500];
	system("mkdir \"imgTeste\"");
	for (int i = 0; i < n; i++) {
		snprintf(buff, 500, "imgTeste/[%d][%d] ", (int) labelsI[i], m);
		for (int j = 0; j < m; j++) {
			snprintf(buff, 500, "%s%.1lf,", buff, labels[i * m + j]);
		}
		snprintf(buff, 500, "%s.ppm", buff);
		ppmp2(images + i * x * y * z, x, y, buff);
	}
}
