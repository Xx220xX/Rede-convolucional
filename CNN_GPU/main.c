#include <conio.h>
#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"

// call backs
void onLoad(ManageTrain *t);


void OnfinishEpic(ManageTrain *t);

void OnInitTrain(ManageTrain *t);

void OnfinishTrain(ManageTrain *t);

void OnInitFitnes(ManageTrain *t);

void OnfinishFitnes(ManageTrain *t);

void UpdateTrain(Estatistica *t);

void UpdateFitnes(ManageTrain *t);

int main(int arg, char **args) {
	system("chcp 65001");
//	printf("##############################\n");
//	printf("Gabriela IA\n");
//	printf("email: gab.cnn.ia@gmail.com\n");
//	printf("Versão %s\n", getVersion());
//	printf("##############################\n");

	if (arg != 2) {
		fprintf(stderr, "É esperado um arquivo lua para iniciar o treinamento");
		return -1;
	}
	ManageTrain manageTrain = createManageTrain(args[1], 0.1, 0.0, 0.0);
	manage2WorkDir(&manageTrain);
//	ManageTrainSetEvent(manageTrain.OnloadedImages, onLoad);
//	ManageTrainSetEvent(manageTrain.OnInitTrain, OnInitTrain);
//	ManageTrainSetEvent(manageTrain.OnfinishEpic, OnfinishEpic);
	ManageTrainSetEvent(manageTrain.UpdateTrain, UpdateTrain);

	// ler imagens
	ManageTrainloadImages(&manageTrain);
	manageTrainLoop(&manageTrain, 0);

	// Treinar
	ManageTraintrain(&manageTrain);
	manageTrainLoop(&manageTrain, 0);


	getClError(manageTrain.cnn->error.error, manageTrain.cnn->error.msg, EXCEPTION_MAX_MSG_SIZE);
	printf("%d %s\n", manageTrain.cnn->error.error, manageTrain.cnn->error.msg);
	releaseManageTrain(&manageTrain);
	return 0;
}

void onLoad(ManageTrain *t) {

	printf("Imagens carregadas com sucesso\n");

//	fflush(stderr);
//	fflush(stdout);
//	char name[100];
//	int len;
//	double *v;
//	char *aux;
//	printf("%u %u %u %u\n",t->imagens->x,t->imagens->y,t->imagens->z,t->imagens->w);
//	for (int w = 0; w < t->imagens->w && w < 300; w++) {
//		len = snprintf(name, 100, "imagens/l%d_%d.", w, (int) ((char *) t->labels->host)[w]);
//		v = t->targets->host + w * t->targets->y * sizeof(double);
//		aux = name;
//		for (int i = 0; i < t->n_classes; i++) {
//			aux = aux + len;
//			len = snprintf(aux, 100 - (size_t) aux + (size_t) name, "%d_", (int) v[i]);
//		}
//		aux = aux + len;
//		len = snprintf(aux, 100 - (size_t) aux + (size_t) name, ".ppm");
//
//		salveTensor4DAsPPM(name, t->imagens, t->cnn, w);
//	}
}

void OnfinishEpic(ManageTrain *t) {
	printf("Epoca: \n");
}

void OnInitTrain(ManageTrain *t) {
	printf("treino iniciado\n");
}

void OnfinishTrain(ManageTrain *t) {
	printf("Treino finalizado\n");
}

void OnInitFitnes(ManageTrain *t) {

}

void OnfinishFitnes(ManageTrain *t) {

}


void UpdateFitnes(ManageTrain *t) {

}

void UpdateTrain(Estatistica *t) {
	double imps = 0;
	if (t->tr_time)
		imps = (t->tr_epoca_atual * t->tr_numero_imagens + t->tr_imagem_atual) / (double) t->tr_time * 1000.0;
	size_t tmp_restante_epoca = (t->tr_numero_imagens - t->tr_imagem_atual) / imps;
	size_t tmp_restante_treino =
			((t->tr_numero_epocas - t->tr_epoca_atual) * t->tr_numero_imagens + t->tr_numero_imagens -
			 t->tr_imagem_atual) / imps;
	printf("epoca %d Imagem %d total %u "
		   "%.lf im/s tempo restante %lld:%02lld:%02lld "
		   "tempo para o fim da epoca %lld:%02lld:%02lld \n",
		   t->tr_epoca_atual,
		   t->tr_imagem_atual,
		   (t->tr_numero_imagens - t->tr_imagem_atual),
		   imps,
		   tmp_restante_epoca / 3600,
		   (tmp_restante_epoca % 3600) / 60,
		   (tmp_restante_epoca % 3600) % 60,
		   tmp_restante_treino / 3600,
		   (tmp_restante_treino % 3600) / 60,
		   (tmp_restante_treino % 3600) % 60

	);

	char c = 0;
	if (kbhit()) {
		c = getche();
		if (c == 'q') {
			manageTrainSetRun((ManageTrain *) t, 0);
		}
	}
}
