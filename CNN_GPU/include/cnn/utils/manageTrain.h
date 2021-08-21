//
// Created by Henrique on 28-Jul-21.
//

#ifndef CNN_GPU_MANAGETRAIN_H
#define CNN_GPU_MANAGETRAIN_H

#include"cnn.h"
#include <stdatomic.h>
#include "Thread.h"
#include "utils/String.h"

typedef void (*ManageEvent)(void *);


typedef struct {
	double *tr_mse_vector;
	double *tr_acertos_vector;
	UINT tr_imagem_atual;
	UINT tr_numero_imagens;
	UINT tr_epoca_atual;
	UINT tr_numero_epocas;
	double tr_erro_medio;
	double tr_acerto_medio;
	size_t tr_time;


	UINT ft_imagem_atual;
	UINT ft_numero_imagens;
	double *ft_info; // Acerto, acerto medio, erro medio
	UINT ft_numero_classes;
	size_t ft_time;
} Estatistica;


typedef struct {
	Estatistica et;
	Cnn cnn;
	// diretorio de trabalho
	String homePath;
	// arquivos de treino
	String file_images;
	String file_labels;

	UINT headers_images;
	UINT headers_labels;

	Tensor imagens;//double
	Tensor targets;//double
	Tensor labels;//char

	// controle de treino
	int n_epics;
	int epic;
	int n_images;
	int n_images2train;
	int n_images2fitness;
	int image;
	int n_classes;
	String class_names;
	char character_sep; // por padrao é ',' e nao pode ser 0x00 (null)

	// estatisticas de treino
	double sum_erro;
	int sum_acerto;


	double current_time;

	// eventos
	ManageEvent OnloadedImages;
	ManageEvent OnfinishEpic;
	ManageEvent OnInitTrain;
	ManageEvent OnfinishTrain;
	ManageEvent OnInitFitnes;
	ManageEvent OnfinishFitnes;

	ManageEvent UpdateTrain;
	ManageEvent UpdateFitnes;
	// controle de memoria
	char self_release;

	// id para thread processo (ler imagens, treinar e avaliar rede)
	char real_time;
	Thread process;
	Thread update_loop;
	// controle do processo
	atomic_int can_run;
	atomic_int process_id;
} ManageTrain;


void manage2WorkDir(ManageTrain *t);

void releaseManageTrain(ManageTrain *t);

void manageTrainSetEvent(ManageEvent *dst, ManageEvent src);

void manageTrainSetRun(ManageTrain *t, int run);

ManageTrain createManageTrain(char *luafile, double tx_aprendizado, double momento, double decaimento);

int ManageTrainloadImages(ManageTrain *t);

int ManageTraintrain(ManageTrain *t);

int ManageTrainfitnes(ManageTrain *t);

void manageTrainLoop(ManageTrain *t, int run_background);

#define ManageTrainSetEvent(dst, src)manageTrainSetEvent(&(dst),(ManageEvent)(src))
#endif //CNN_GPU_MANAGETRAIN_H
