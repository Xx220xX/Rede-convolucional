//
// Created by Henrique on 02/12/2021.
//

#ifndef GAB_SETUP_H
#define GAB_SETUP_H

#include "cnn/cnn_lua.h"

#define Free(string)if(string)gab_free(string);(string)= NULL
#define Swap(a, b)do{typeof(a) _local__aux__swap_ = a;(a) = b;    (b) = __local__aux__swap__;}while(0)

typedef char *String;
typedef struct {
	uint32_t imAtual;
	uint32_t imTotal;
	uint32_t epAtual;
	uint32_t epTotal;
	uint32_t totalImages;
	uint32_t imagensCalculadas;

	double mse;
	double faval;
	double winRate;
	double winRateMedio;
	double winRateMedioep;
	double t0;
} Itrain;
typedef struct {
	uint32_t imAtual;
	uint32_t imTotal;
	double mse;
	double winRate;
	double meanwinRate;
	double timeRuning;
} Iteste;
typedef struct {
	uint32_t imAtual;
	uint32_t imTotal;
	double t0;
} ILoad;
typedef struct {
	int classe;
	char name[50];
	int acertos;
	uint32_t *answers;
} TesteClasseEstatisticas;
typedef struct {
	TesteClasseEstatisticas *tce;
	int nclasses;
} TesteEstatisticas;
typedef struct Setup_t {
	// #### nome
	String nome;
	// #### diretorio
	String home;
	// ###### Imagens e Etiquetas
	int use_gpu;
	int useBatch;

	uint32_t n_classes;
	String nome_classes;
	uint32_t n_imagens;
	Tensor *imagens;
	Tensor *targets;
	Tensor labels;
	String file_label;
	String file_image;
	uint32_t header_image;
	uint32_t header_label;

	// ##### Controle de loops
	atomic_int can_run;
	atomic_int runing;
	atomic_int force_end;

	// ##### Configurações do treino
	size_t batchSize;
	size_t batch;
	uint32_t n_epocas;
	uint32_t n_imagens_treinar;
	uint32_t epoca_atual;
	uint32_t imagem_atual_treino;

	// ##### Configurações do teste
	uint32_t n_imagens_testar;
	uint32_t imagem_atual_teste;
	TesteEstatisticas te;

	// informações para serem lidas em outras threads
	String treino_out;
	String teste_out;
	String rede_out;
	ILoad iLoad;
	Itrain itrain;
	Iteste iteste;


	/// Rede neural
	Cnn cnn;

	void (*on_endEpoca)(const struct Setup_t *self, int label);

	// ##### Métodos
	void (*loadImagens)(struct Setup_t *self);

	void (*loadLua)(struct Setup_t *self, const char *lua_file);

	void (*loadLabels)(struct Setup_t *self);

	int (*ok)(struct Setup_t *self);

	void (*treinar)(struct Setup_t *self);

	void (*treinarBatch)(struct Setup_t *self);

	void (*avaliar)(struct Setup_t *self);

	void (*fast_fitnes)(struct Setup_t *self,float *winRate,float *custo);

	void (*saveStatistic)(struct Setup_t *self);

	int (*release)(struct Setup_t **self);

	void (*checkStop)(struct Setup_t *self, const char *stopPattern);

} *Setup, Setup_t;

Setup Setup_new();

char *asprintf(size_t *tlen, const char *format, ...);

double seconds();

#endif //GAB_SETUP_H
