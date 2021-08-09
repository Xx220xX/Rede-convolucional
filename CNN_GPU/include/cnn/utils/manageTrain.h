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
	double *tr_mse_vector;
	double *tr_acertos_vector;
	UINT tr_imagem_atual;
	UINT tr_numero_imagens;
	UINT tr_epoca_atual;
	UINT tr_numero_epocas;
	double tr_erro_medio;
	double tr_acerto_medio;

	UINT ft_imagem_atual;
	UINT ft_numero_imagens;
	double * ft_info; // Acerto, acerto medio, erro medio
	UINT ft_numero_classes;

} Estatistica;


typedef struct {
	Estatistica et;
	// rede convolucional
	Cnn cnn;
	// diretorio de trabalho
	char *homePath;
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

void manage2WorkDir(ManageTrain *t);

void releaseManageTrain(ManageTrain *t);

void manageTrainSetEvent(ManageEvent *dst,ManageEvent src);

void manageTrainSetRun(ManageTrain *t,int run);

ManageTrain createManageTrain(char *luafile,double tx_aprendizado,double momento,double decaimento);

void ManageTrainInitThreadHigh(ManageTrain *t );
#define releaseSTRManageTrain(t, str)if((t).releaseStrings){if((str))free_mem(str);(str)=NULL;}
#endif //CNN_GPU_MANAGETRAIN_H
