#ifndef CNN_GPU_CNN_H
#define CNN_GPU_CNN_H

#include "camadas/Camada.h"
#include "camadas/CamadaConv.h"
#include "camadas/CamadaConvNC.h"
#include "camadas/CamadaRelu.h"
#include "camadas/CamadaPadding.h"
#include "camadas/CamadaDropOut.h"
#include "camadas/CamadaFullConnect.h"
#include "camadas/CamadaPool.h"
#include "camadas/CamadaPoolAv.h"
#include "camadas/CamadaSoftMax.h"
#include "camadas/CamadaBatchNorm.h"

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
	double normaErro;
	void *L;
	fv releaseL;
	Exception error;

} *Cnn, TypeCnn;

///Cria uma Cnn
Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz);

/// Libera os recursos alocados pela Cnn
void releaseCnn(Cnn *pc);

/// Cria uma Cnn a partir de um kernel em um arquivo
Cnn createCnnWithWrapperFile(char *kernelFile, Params p, UINT inx, UINT iny, UINT inz,
                             unsigned long long int devicetype);


/// Cria uma Cnn a partir de um kernel em uma string
Cnn createCnnWithWrapperProgram(const char *kernelprogram, Params p, UINT inx, UINT iny,
                                UINT inz, ULL devicetype);

/// Calcula o erro gerada na saida da rede
int CnnCalculeError(Cnn c);


int Convolucao(Cnn c, char tensor_flag, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT numeroDeFiltros);

int ConvolucaoNcausal(Cnn c, char tensor_flag, UINT passox, UINT passoy, UINT filtrox, UINT filtroy,
                      UINT largx, UINT largy,
                      UINT numeroDeFiltros);

int Pooling(Cnn c, char tensor_flag, UINT passox, UINT passoy,
            UINT filtrox, UINT filtroy);

int PoolingAv(Cnn c, char tensor_flag, UINT passox, UINT pasoy, UINT fx, UINT fy);

int Relu(Cnn c, char tensor_flag);

int Padding(Cnn c, char tensor_flag, UINT top, UINT bottom, UINT left, UINT right);

int BatchNorm(Cnn c, char tensor_flag, double epsilon);

int SoftMax(Cnn c, char tensor_flag);

int Dropout(Cnn c, char tensor_flag, double pontoAtivacao, long long int seed);

int FullConnect(Cnn c, char tensor_flag, UINT tamanhoDaSaida, int funcaoDeAtivacao);

int CnnCall(Cnn c, double *input);

int CnnLearn(Cnn c, double *target);

void cnnSave(Cnn c, FILE *dst);

int cnnCarregar(Cnn c, FILE *src);


void normalizeGPU(Cnn c, double *input, double *output, int len, double maximo, double minimo);

void normalizeGPUSpaceKnow(Cnn c, double *input, double *output, int len, double input_maximo,
                           double input_minimo,
                           double maximo, double minimo);


int CnnGetIndexMax(Cnn c);

void printCnn(Cnn c);

char *salveCnnOutAsPPMGPU(Cnn c, size_t *h_r, size_t *w_r);

const char *getVersion();

const char *getInfo();

#endif