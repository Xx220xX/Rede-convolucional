//
// Created by Henrique on 28-Jul-21.
//

#ifndef CNN_GPU_MANAGETRAIN_H
#define CNN_GPU_MANAGETRAIN_H

#include "cnn/cnn_lua.h"
#include <stdatomic.h>
#include "Thread.h"

#define MANAGE_DEBUG_LOAD_LUA 0

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
	double tr_imps;
	double tr_time;


	UINT ft_imagem_atual;
	UINT ft_numero_imagens;
	double *ft_info; // Acerto, acerto medio, erro medio
	int ft_info_coluns;
	UINT ft_numero_classes;
	size_t ft_time;
	size_t ll_imagem_atual;
	size_t ld_imagem_atual;
} Estatistica;


typedef struct {
	Estatistica et;
	Cnn cnn;
	// diretorio de trabalho
	char *homePath;
	char *name;
	// arquivos de treino
	char *file_images;
	char *file_labels;
	// saida
	char *treino_info;
	char *fitnes_info;

	UINT headers_images;
	UINT headers_labels;

	Tensor *imagens;//REAL
	Tensor *targets;//REAL
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
	char use_gpu_mem;


	int sum_acerto;

	double current_time;
	// eventos
	ManageEvent OnfinishEpic;


	ManageEvent UpdateTrain;
	ManageEvent UpdateFitnes;
	ManageEvent UpdateLoad;

	// controle de memoria
	char self_release;
	char real_time;
	// id para thread processo (ler imagens, treinar e avaliar rede)
	HANDLE process;
	HANDLE update_loop;
	// controle do processo
	atomic_int can_run;
	atomic_int force_close;
	atomic_int process_id;

} Manage;


void Manage_to_workDir(Manage *t);

double getus();

void Manage_release(Manage *t);

void Manage_setEvent(ManageEvent *dst, ManageEvent src);

void Manage_run(Manage *t, int run);

void Manage_update_lua_params(Manage *manage);

Manage Manage_new(char *luafile, int luaIsProgram);

int Manage_load(Manage *t, int runBackground);

int Manage_train(Manage *t, int runBackground);

int Manage_fitnes(Manage *t, int runBackground);

void Manage_loop(Manage *t, int run_background);

#define ManageTrainSetEvent(dst, src)Manage_setEvent(&(dst),(ManageEvent)(src))

#endif //CNN_GPU_MANAGETRAIN_H
