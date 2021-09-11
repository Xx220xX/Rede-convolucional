#ifndef CNN_GPU_CNN_H
#define CNN_GPU_CNN_H

#include "config.h"
#include "camadas/Camada.h"
#include "camadas/CamadaConv.h"
#include "camadas/CamadaConvNC.h"
#include "camadas/CamadaRelu.h"
#include "camadas/CamadaPRelu.h"
#include "camadas/CamadaPadding.h"
#include "camadas/CamadaDropOut.h"
#include "camadas/CamadaFullConnect.h"
#include "camadas/CamadaPool.h"
#include "camadas/CamadaPoolAv.h"
#include "camadas/CamadaSoftMax.h"
#include "camadas/CamadaBatchNorm.h"

#include "utils/list_args.h"

///Armazena os dados de uma rede neural convolucional
typedef struct Cnn {
	Params parametros;
	Camada *camadas;
	Tensor lastGrad;
	Tensor target;
	int size;
	Ponto sizeIn;
	QUEUE queue;
	WrapperCL *cl;
	char releaseCL;

	Kernel kernelsub;
	Kernel kerneldiv;
	Kernel kerneldivInt;
	Kernel kernelNormalize;
	Kernel kernelInt2Vector;
	Kernel kernelcreateIMG;

	void *L;
	Dictionary luaArgs;
	fv releaseL;
	CNN_ERROR error;

	char release_self;
} *Cnn, Cnn_t;

///Cria uma Cnn
Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz);

/// Libera os recursos alocados pela Cnn
void releaseCnn(Cnn *pc);

void CnnRemoveLastLayer(Cnn c);

/// Cria uma Cnn a partir de um kernel em um arquivo
Cnn createCnnWithWrapperFile(const char *kernelFile, Params p, UINT inx, UINT iny, UINT inz,
							 uint64_t devicetype);


/// Cria uma Cnn a partir de um kernel em uma string
Cnn createCnnWithWrapperProgram(const char *kernelprogram, Params p, UINT inx, UINT iny,
								UINT inz, ULL devicetype);

/// Calcula o erro gerada na saida da rede
int CnnCalculeError(Cnn c, double *mse);

int CnnCalculeErrorWithOutput(Cnn c, double *target, double *mse);

int CnnCalculeErrorTWithOutput(Cnn c, Tensor target, double *mse);

int CnnGetIndexMax(Cnn c);

int Convolucao(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT numeroDeFiltros);

int ConvolucaoNcausal(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT largx, UINT largy, UINT numeroDeFiltros);

int Pooling(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy);

int PoolingAv(Cnn c, UINT passox, UINT pasoy, UINT fx, UINT fy);

int Relu(Cnn c);

int PRelu(Cnn c);

int Padding(Cnn c, UINT top, UINT bottom, UINT left, UINT right);

int BatchNorm(Cnn c, double epsilon);

int SoftMax(Cnn c);

int Dropout(Cnn c, double pontoAtivacao, long long int seed);

int FullConnect(Cnn c, UINT tamanhoDaSaida, int funcaoDeAtivacao);

int CnnCall(Cnn c, double *input);

int CnnCallT(Cnn c, Tensor input);

int CnnLearn(Cnn c, double *target);

int CnnLearnT(Cnn c, Tensor target);

void CnnInitLuaVm(Cnn c);

int CnnLuaConsole(Cnn c);

void LuaputHelpFunctionArgs(void (*myf)());

int CnnLuaLoadFile(Cnn c, const char *file_name);

void cnnSave(Cnn c, FILE *dst);


int cnnCarregar(Cnn c, FILE *src);

void normalizeGPU(Cnn c, double *input, double *output, int len, double maximo, double minimo);


void normalizeGPUSpaceKnow(Cnn c, double *input, double *output, int len, double input_maximo,
						   double input_minimo,
						   double maximo, double minimo);

void printCnn(Cnn c);

char *salveCnnOutAsPPMGPU(Cnn c, size_t *h_r, size_t *w_r);

const char *getVersion();

void showVersion();


#endif