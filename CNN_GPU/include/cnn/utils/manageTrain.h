//
// Created by Henrique on 28-Jul-21.
//

#ifndef CNN_GPU_MANAGETRAIN_H
#define CNN_GPU_MANAGETRAIN_H

#include"cnn.h"
#include <stdatomic.h>
#include "pthread.h"

typedef void (*ManageEvent)(void *);

typedef struct {
	double *erros;
	double *acertos;
	int n;
	int max_size;
} Estatistica;

typedef struct {
	// rede convolucional
	Cnn cnn;

	// arquivos de treino
	char *file_images;
	char *file_labels;

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
	int image;
	int n_classes;
	char *class_names;
	char character_sep; // por padrao é ',' e nao pode ser 0x00 (null)

	// estatisticas de treino
	double sum_erro;
	int sum_acerto;
	double mean_error;
	int mean_hit;
	Estatistica et;

	double current_time;

	// eventos
	ManageEvent OnloadedImages;
	ManageEvent OnfinishEpic;
	ManageEvent OnInitTrain;
	ManageEvent OnfinishTrain;
	ManageEvent finishFitnes;

	// controle do processo
	atomic_int can_run;

	// id para thread
	pid_t process;
} ManageTrain;

void loadImages(ManageTrain *t);

void train(ManageTrain *t);

void fitnes(ManageTrain *t);

#endif //CNN_GPU_MANAGETRAIN_H
