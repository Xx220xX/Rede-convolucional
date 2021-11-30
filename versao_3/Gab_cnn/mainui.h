#include <conio.h>
#include <windows.h>
#include <dir.h>
#include <math.h>
#include "treino/Manage.h"
#include "conio2/conio2.h"
#include "ui.h"
// call backs

void onLoad(Manage *t);

void OnfinishEpic(Manage *t);

void OnInitTrain(Manage *t);

void OnfinishFitnes(Manage *t);

void UpdateTrain(Manage *t);

void UpdateFitnes(Manage *t);

void onLoad(Manage *t) {
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

void OnfinishEpic(Manage *t) {
//	printf("Epoca: \n");
	DVector *v = alloc_mem(sizeof(DVector), 1);

	snprintf(v->file, 250, "statistic/%d.bin", t->epic);
	v->self_release = 1;
	v->releasex_y = 1;
	v->x = t->et.tr_mse_vector;
	v->y = t->et.tr_acertos_vector;
	v->length = t->et.tr_imagem_atual + 1;
	t->et.tr_mse_vector = alloc_mem(t->n_images, sizeof(double));
	t->et.tr_acertos_vector = alloc_mem(t->n_images, sizeof(double));

	HANDLE th = Thread_new(saveAsGraphic, v);
	ThreadClose(th);
}

void OnInitTrain(Manage *t) {
//    printf("treino iniciado\n");

}

void OnfinishFitnes(Manage *t) {
	FILE *f = fopen("tabela/fitnes.csv", "w");
	fprintf(f, "Classe,Casos,Número de acertos,Taxa de acerto,Erro médio");
	for (int k = 0; k < t->n_classes; k++) {
		fprintf(f, ",classe %d", k + 1);
	}
	fprintf(f, "\n");
	char *msg = t->class_names;
	int i = 0;
	char sep = t->character_sep;
	sep = ' ';
	REAL media = 0;
	size_t class_namessize = strlen(t->class_names);
	for (int j = 0; j < t->n_classes; ++j) {
		for (; i <class_namessize && msg[i] != sep; i++) {
			if (msg[i] == ',')continue;
			fprintf(f, "%c", msg[i]);
		}
		fprintf(f, ",");
		fprintf(f, "%d,%d,%lf,%lf", (int) t->et.ft_info[0 + j * t->et.ft_info_coluns], (int) t->et.ft_info[1 + j * t->et.ft_info_coluns],
				100 * t->et.ft_info[1 + j * t->et.ft_info_coluns] / (t->et.ft_info[0 + j * t->et.ft_info_coluns] + 1e-14),
				t->et.ft_info[2 + j * t->et.ft_info_coluns] / (t->et.ft_info[0 + j * t->et.ft_info_coluns] + 1e-14));
		for (int k = 0; k < t->n_classes; k++) {
			if (k == j)
				fprintf(f, ",-");
			else
				fprintf(f, ",%d", (int) t->et.ft_info[3 + k + j * t->et.ft_info_coluns]);
		}
		fprintf(f, "\n");
		media += t->et.ft_info[1 + j * t->et.ft_info_coluns];
		i++;
	}
	fprintf(f, "Media de acerto,%lf\n", 100 * media / t->et.ft_imagem_atual);
	fclose(f);
	f = fopen("showGraphic.py", "w");
	{
		fprintf(f, "dir = './statistic/'\n"
				   "import os, ctypes as c\n"
				   "import matplotlib.pyplot as plt\n"
				   "import numpy as np\n"
				   "files = [dir + x for x in os.listdir(dir)]\n"
				   "geralx = []\n"
				   "geraly = []\n"
				   "geralt = []\n"
				   "i = 0\n"
				   "for file in files:\n"
				   "\tif not file.endswith('.bin'): continue\n"
				   "\ti+=1\n"
				   "\t#if i == 30:break\n"
				   "\twith open(file, 'rb') as f:\n"
				   "\t\tb = f.read(8)\n"
				   "\t\tsize_t = c.c_char * 8\n"
				   "\t\tb = size_t(*b)\n"
				   "\t\tlength = c.cast(b, c.POINTER(c.c_size_t))[0]\n"
				   "\t\tx = f.read(length * 8)\n"
				   "\t\ty = f.read(length * 8)\n"
				   "\t\tv_c = c.c_char * (length * 8)\n"
				   "\t\tx = c.cast(v_c(*x), c.POINTER(c.c_double))\n"
				   "\t\ty = c.cast(v_c(*y), c.POINTER(c.c_double))\n"
				   "\t\tx = [x[i] for i in range(1000,length)]\n"
				   "\t\ty = [y[i] for i in range(1000,length)]\n"
				   "\t\t#plt.figure(file)\n"
				   "\n"
				   "\t\t#plt.title(file.replace(dir, 'epoca ').replace('.bin', ''))\n"
				   "\t\t#plt.plot(x)\n"
				   "\t\t#plt.plot(y)\n"
				   "\t\t#plt.legend(['erro medio', 'acerto medio'])\n"
				   "\n"
				   "\t\tgeralx.extend(x)# sum(x) / len(x))\n"
				   "\t\tgeraly.extend(y)#sum(y) / len(y))\n"
				   "\t\tt0 = 0\n"
				   "\t\tif len(geralt)>0:t0 = geralt[-1]\n"
				   "\t\tgeralt.extend(list(t0+np.arange(0,1,1/len(y))))#(len(geralt) + 1)\n"
				   "plt.figure('Final')\n"
				   "plt.title('Aprendizado')\n"
				   "plt.plot(geralt, geralx)\n"
				   "plt.plot(geralt, geraly)\n"
				   "plt.xlabel('epoca')\n"
				   "plt.legend(['Erro', 'Taxa Acerto'])\n"
				   "\n"
				   "plt.show()\n"
				   "");
	}
	fflush(f);
	fclose(f);
}

void UpdateTrain(Manage *mt) {
	double imps = 0;
	Estatistica *t = (Estatistica *) mt;
//	if (t->tr_time)
//		imps = (t->tr_epoca_atual * t->tr_numero_imagens + t->tr_imagem_atual) / (REAL) t->tr_time * 1000.0;
	imps = (double) t->tr_imps;
	size_t tmp_restante_epoca = round((t->tr_numero_imagens - t->tr_imagem_atual - 1) / imps);
	size_t tmp_restante_treino =
			round(((t->tr_numero_epocas - t->tr_epoca_atual - 1) * t->tr_numero_imagens + t->tr_numero_imagens -
				   t->tr_imagem_atual - 1) / imps);
	static int y = -1;
	static Progress pepoca = {0};
	static Progress pimagen = {0};
	if (y == -1) {
		y = wherey() + 3;
		pepoca = newProgress("Epocas", 0, 1, y, 30);
		pimagen = newProgress("Imagens", 0, 1, y + 1, 30);

	}

	showP(&pepoca, (double) (t->tr_epoca_atual * t->tr_numero_imagens + t->tr_imagem_atual) / ((double) t->tr_numero_epocas * t->tr_numero_imagens));
	showP(&pimagen, (double) (t->tr_imagem_atual + 1.0) / (t->tr_numero_imagens));
	pimagen.end = 0;
	printf("\n");
	printf("Tempo estimado final do treino  %lld:%02lld:%02lld\n",
		   tmp_restante_treino / 3600,
		   (tmp_restante_treino % 3600) / 60,
		   (tmp_restante_treino % 3600) % 60);
	printf("Tempo estimado final da epoca %lld:%02lld:%02lld\n",
		   tmp_restante_epoca / 3600,
		   (tmp_restante_epoca % 3600) / 60,
		   (tmp_restante_epoca % 3600) % 60);
	printf("Imagens por segundo %.2lf\n",imps);
	printf("Mse %.16lf\n"
		   "Acerto medio %.2lf     \n",
		   t->tr_erro_medio,
		   t->tr_acerto_medio * 100);



	char c = 0;
	if (kbhit()) {
		c = getche();
		if (c == 'q') {
			Manage_run((Manage *) t, 0);
		}
	}
	Sleep(10);
}

void UpdateFitnes(Manage *t) {
	static int y = -1;
	if (y == -1) {
		y = wherey() + 2;
	}
	gotoxy(1, y);
	printf("Imagem %d de %d   %lf%%    ", t->et.ft_imagem_atual + 1, t->et.ft_numero_imagens, (t->et.ft_imagem_atual + 1.0) / t->et.ft_numero_imagens * 100);
	double imps = 1e-40;
	double tm = t->et.ft_time * 1e-3;
	if (tm != 0)
		imps = (t->et.ft_imagem_atual + 1) / tm;
	size_t tmp = (t->et.ft_numero_imagens - t->et.ft_imagem_atual - 1.0) / imps;
//		printf("Tempo estimado final da avaliação --:--:--     \n");
	printf("\n%lf tr_imps     ", (double) imps);
//	printf("\n%lf temp     ",tm);
	printf("\nTempo estimado final da avaliação %lld:%02lld:%02lld     \n",
		   tmp / 3600,
		   (tmp % 3600) / 60,

		   (tmp % 3600) % 60);
	char c = 0;
	if (kbhit()) {
		c = getche();
		if (c == 'q') {
			Manage_run((Manage *) t, 0);
		}
	}
}


void UpdateLoad(Manage *t) {
	static Progress pi = {0};
	static Progress pl = {0};
	static int y = -1;
	if (y == -1) {
		y = wherey() + 1;
		pi = newProgress("Imagem ", 0, 1, y, 20);
		pl = newProgress("Label ", 0, 1, y + 1, 20);
	}

	showP(&pi, (t->et.ld_imagem_atual + 1.0) / t->n_images);
	showP(&pl, (t->et.ll_imagem_atual + 1.0) / t->n_images);
}
