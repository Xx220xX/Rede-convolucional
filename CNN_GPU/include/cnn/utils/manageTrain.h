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
	Tensor fitness_hit_rate;
	UINT image_fitnes;
	UINT max_size;
	UINT image;
	UINT epic;
	double mean_error;
	double hit_rate;
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
	int n_images2fitness;
	int image;
	int n_classes;
	char *class_names;
	char character_sep; // por padrao é ',' e nao pode ser 0x00 (null)

	// estatisticas de treino
	double sum_erro;
	int sum_acerto;

	Estatistica et;

	double current_time;

	// eventos
	ManageEvent OnloadedImages;
	ManageEvent OnfinishEpic;
	ManageEvent OnInitTrain;
	ManageEvent OnfinishTrain;
	ManageEvent OnInitFitnes;
	ManageEvent OnfinishFitnes;

	// controle do processo
	atomic_int can_run;

	// id para thread
	pid_t process;
	// controle de memoria
	char self_release;
	char releaseStrings;
} ManageTrain;

void loadImages(ManageTrain *t);

void train(ManageTrain *t);

void fitnes(ManageTrain *t);

void releaseManageTrain(ManageTrain *t);

#endif //CNN_GPU_MANAGETRAIN_H
