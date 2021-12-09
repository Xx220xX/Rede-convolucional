#include "windows.h"
#include <stdio.h>
#include <cnn/cnn_lua.h>
#include <png/png.h>
#include "ui.h"
#include "setup/Setup.h"
#include "conio2/conio2.h"
#include "thread/Thread.h"

void Cnn_asimage(Cnn c, char *file, int largura, int altura, ...) {
	Tensor s = NULL;
	va_list v;
	uint8_t *image = alloc_mem(altura, largura);
	va_start(v, altura);
	s = va_arg(v, Tensor);
	int nTensors = 0;
	while (s) {
		nTensors++;
		s = va_arg(v, Tensor);
	}
	va_end(v);
	va_start(v, altura);
	int h = altura / nTensors;
	int w;
	int sz;
	int k;
	for (int j = 0; j < nTensors; ++j) {
		s = va_arg(v, Tensor);
		w = largura / (s->z * s->w);
		sz = h<w?h:w;
		for (int l = 0; l < s->w; ++l) {
			for (int z = 0; z < s->z; ++z) {
				k = l * s->z + z;
				if(s->x != 1) {
					s->imagegray(s, image, largura, altura, sz, sz, j * h, k * w, z, l);
				}else{
					s->imagegray(s, image, largura, altura, w, h, j * h, k * w, z, l);

				}
			}
		}
	}
	va_end(v);
//	FILE *f = fopen(file,"wb");
//	if(f) {
		pngGRAY(file, image, largura, altura);
//		fclose(f);
//	}else{
//		fprintf(stderr,"erro %d: Falha ao abrir arquivo %s\n",FAILED_OPEN_FILE,file);
//	}
	free_mem(image);
}

void on_train(const struct Setup_t *self,int label) {
	Cnn cnn = self->cnn;
	String file = asprintf(NULL, "imagens/%d/sample_%d.png",label, self->imagem_atual_treino);
#define TSM super.s
	Cnn_asimage(cnn, file, 640, 640,
				cnn->cm[0]->a,
				cnn->cm[0]->s,
				cnn->cm[1]->s,
				cnn->cm[2]->s,
				cnn->cm[3]->s,
				cnn->cm[4]->s,
				NULL);
	Free(file);
}

int cnnMain(int nargs, char **args) {
	LCG_setSeed(time(0));

	Setup s = Setup_new();
	s->on_train = on_train;
	GUI.can_run = &s->can_run;
	GUI.force_end = &s->force_end;
	char luaFile[250] = {0};
	getLuaFILE(luaFile, 250, nargs, (const char **) args);
	s->loadLua(s, luaFile);
	String cnn_description = s->cnn->printstr(s->cnn, "->");
	GUI.setText(GUI.rede, cnn_description);
	Free(cnn_description);

	if (s->ok(s)) {
		double t0 = seconds();
		s->runing = 1;
		Handle hload = Thread_new(s->loadImagens, s);
		while (s->runing) {
			GUI.setText(GUI.status, "Carregando Imagens: %d de %d  \n", s->iLoad.imAtual, s->iLoad.imTotal);
			GUI.setProgress(GUI.progress, s->iLoad.imAtual * 100.0 / s->iLoad.imTotal);
			Sleep(100);
		}
		GUI.setProgress(GUI.progress, s->iLoad.imAtual * 100.0 / s->iLoad.imTotal);
		Thread_Release(hload);
		t0 = seconds() - t0;
		printf("Tempo para leitura de imagens %.3lf s\n",t0);
	}
	if (s->ok(s)) {
		double t0 = seconds();
		s->runing = 1;
		HANDLE hload = Thread_new(s->loadLabels, s);
		while (s->runing) {
			GUI.setText(GUI.status, "Carregando labels: %d de %d        \n", s->iLoad.imAtual, s->iLoad.imTotal);
			GUI.setProgress(GUI.progress, s->iLoad.imAtual * 100.0 / s->iLoad.imTotal);
			Sleep(100);
		}
		GUI.setProgress(GUI.progress, s->iLoad.imAtual * 100.0 / s->iLoad.imTotal);
		showCursor(1);
		Thread_Release(hload);
		t0 = seconds() - t0;
		printf("Tempo para leitura de labels %.3lf s\n",t0);
	}

	double im;
	if (s->ok(s)) {
		double t0 = seconds();
		s->runing = 1;
		Handle htreino = Thread_new(s->treinar, s);
		Itrain treino;
		GUI.setText(GUI.status, "Treinando");
		while (s->runing) {
			treino = s->itrain;
			im = treino.imAtual + treino.epAtual * treino.imTotal;
			GUI.setProgress(GUI.progress, im * 100.0 / (treino.imTotal * treino.epTotal));
			GUI.setText(GUI.epoca, "%d de %d", treino.epAtual, treino.epTotal);
			GUI.setText(GUI.imagem, "%d de %d", treino.imAtual, treino.imTotal);
			GUI.setText(GUI.mse, "%.14lf", treino.mse);
			GUI.setText(GUI.winHate, "%lf", treino.winRate);
			
			GUI.setText(GUI.imps, "%.2lf | %.2lf ", treino.imps,im/treino.timeRuning);
			Sleep(100);
		}
		Thread_Release(htreino);
		t0 = seconds() - t0;
		printf("Tempo para treino %.3lf s\n",t0);
	}
	char *tmp = asprintf(NULL, "MSE %lf\nwin hate %lf.\n", s->itrain.mse, s->itrain.winRate);
	dialogBox("Treino terminado!", tmp);
	free_mem(tmp);
	return s->release(&s);
}
