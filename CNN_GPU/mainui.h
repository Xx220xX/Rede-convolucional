#include <conio.h>
#include <windows.h>
#include <utils/dir.h>
#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include "conio2/conio2.h"

// call backs
void onLoad(ManageTrain *t);

void OnfinishEpic(ManageTrain *t);

void OnInitTrain(ManageTrain *t);

void OnfinishTrain(ManageTrain *t);

void OnInitFitnes(ManageTrain *t);

void OnfinishFitnes(ManageTrain *t);

void UpdateTrain(ManageTrain *t);

void UpdateFitnes(ManageTrain *t);

void onLoad(ManageTrain *t) {
//    printf("Imagens carregadas com sucesso\n");
//    Sleep(400);
}

typedef struct {
	double *x, *y;
	size_t length;
	char file[250];
	char self_release;
	char releasex_y;
} DVector;

int saveAsGraphic(DVector *a) {
	FILE *f = fopen(a->file, "wb");
	fwrite(&a->length, sizeof(size_t), 1, f);

	fwrite(a->x, sizeof(double), a->length, f);
	fwrite(a->y, sizeof(double), a->length, f);
	fflush(f);
	fclose(f);
	if (a->releasex_y) {
		free_mem(a->x);
		free_mem(a->y);
	}
	if (a->self_release) {
		free_mem(a);
	}
	return 0;
}

void OnfinishEpic(ManageTrain *t) {
//	printf("Epoca: \n");
	DVector *v = alloc_mem(sizeof(DVector), 1);
	createDir("statistic");
	snprintf(v->file, 250, "statistic/%d.bin", t->epic);
	v->self_release = 1;
	v->releasex_y = 1;
	v->x = t->et.tr_mse_vector;
	v->y = t->et.tr_acertos_vector;
	v->length = t->et.tr_imagem_atual + 1;
	t->et.tr_mse_vector = alloc_mem(t->n_images, sizeof(double));
	t->et.tr_acertos_vector = alloc_mem(t->n_images, sizeof(double));

	Thread th = newThread(saveAsGraphic, v, NULL);
	ThreadClose(th);
}

void OnInitTrain(ManageTrain *t) {
//    printf("treino iniciado\n");

}

void OnfinishFitnes(ManageTrain *t) {
	FILE *f = fopen("fitnes.csv", "w");
	fprintf(f, "Classe,Casos,Taxa de acerto,Erro médio\n");
	char *msg = t->class_names.d;
	int i = 0;
	char sep = t->character_sep;
	sep =  ' ';
	for (int j = 0; j < t->n_classes; ++j) {
		for (; i < t->class_names.size && msg[i] != sep; i++) {
			if (msg[i] == ',')continue;
			fprintf(f, "%c", msg[i]);
		}
		fprintf(f, ",");
		fprintf(f, "%d,%lf,%lf\n", (int) t->et.ft_info[0 + j * 3],100* t->et.ft_info[1 + j * 3] / (t->et.ft_info[0 + j * 3] + 1e-14), t->et.ft_info[2 + j * 3] / (t->et.ft_info[0 + j * 3] + 1e-14));
		i++;
	}
	fclose(f);
}

void UpdateTrain(ManageTrain *mt) {
	double imps = 0;
	Estatistica *t = (Estatistica *) mt;
	if (t->tr_time)
		imps = (t->tr_epoca_atual * t->tr_numero_imagens + t->tr_imagem_atual) / (double) t->tr_time * 1000.0;
	size_t tmp_restante_epoca = round((t->tr_numero_imagens - t->tr_imagem_atual - 1) / imps);
	size_t tmp_restante_treino =
			round(((t->tr_numero_epocas - t->tr_epoca_atual - 1) * t->tr_numero_imagens + t->tr_numero_imagens -
				   t->tr_imagem_atual - 1) / imps);
	static int y = -1;
	if (y == -1) {
		y = wherey() + 2;
	}
	gotoxy(1, y);
	printf("Epoca %d de %d     imagem %d de %d \n", t->tr_epoca_atual, t->tr_numero_epocas, t->tr_imagem_atual, t->tr_numero_imagens);
	printf("Tempo estimado final do treino  %lld:%02lld:%02lld\n",
					tmp_restante_treino / 3600,
					(tmp_restante_treino % 3600) / 60,
					(tmp_restante_treino % 3600) % 60);
	printf("Tempo estimado final da epoca %lld:%02lld:%02lld\n",
					tmp_restante_epoca / 3600,
					(tmp_restante_epoca % 3600) / 60,
					(tmp_restante_epoca % 3600) % 60);
	printf("Imagens por segundo %.2lf\n",
					imps);
	printf("Mse %.16lf\n"
					"Acerto medio %lf\n",
					t->tr_erro_medio,
					t->tr_acerto_medio * 100);

	static int delete = 0;
	char c = 0;
	if (kbhit()) {
		c = getche();
		if (c == 'q') {
			manageTrainSetRun((ManageTrain *) t, 0);
		}
	}
	Sleep(1);
//	system("cls");
}

void UpdateFitnes(ManageTrain *t) {
	static int y = -1;
	if (y == -1) {
		y = wherey() + 2;
	}
	gotoxy(1, y);
	printf("Imagem %d de %d   %lf%%    ",t->et.ft_imagem_atual+1,t->et.ft_numero_imagens,(t->et.ft_imagem_atual+1.0)/t->et.ft_numero_imagens*100);
	double imps = 1e-40;
	double tm = t->et.ft_time *1e-3;
	if(tm!=0)
		imps = (t->et.ft_imagem_atual+1)/tm;
	size_t tmp  = (t->et.ft_numero_imagens - t->et.ft_imagem_atual - 1.0)/imps;
//		printf("Tempo estimado final da avaliação --:--:--     \n");
	printf("\n%lf imps     ",imps);
//	printf("\n%lf temp     ",tm);
	printf("\nTempo estimado final da avaliação %lld:%02lld:%02lld     \n",
		   tmp / 3600,
		   (tmp % 3600) / 60,
		   (tmp % 3600) % 60);
	char c = 0;
	if (kbhit()) {
		c = getche();
		if (c == 'q') {
			manageTrainSetRun((ManageTrain *) t, 0);
		}
	}
}

void UpdateLoad(ManageTrain *t) {
	static int y = -1;
	if (y == -1) {
		y = wherey() + 1;
	}
	gotoxy(1, y);
	printf("Imagem: %lld   %lf%%      \n", t->et.ld_imagem_atual+1, 100.0*(t->et.ld_imagem_atual+1.0)/t->n_images);
	printf("Label: %lld    %lf%%        ", t->et.ll_imagem_atual+1,100.0*(t->et.ll_imagem_atual+1.0)/t->n_images );
}
