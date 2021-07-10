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


#define INVALID_FILTER_SIZE (-71)
typedef struct {
	Params parametros;
	Camada *camadas;
	Tensor lastGrad;
	Tensor target;
	int size;
	Ponto3d sizeIn;
	cl_command_queue queue;
	WrapperCL *cl;
	char releaseCL;
	GPU_ERROR error;
	Kernel kernelsub;
	Kernel kerneldiv;
	Kernel kerneldivInt;
	Kernel kernelNorm;
	Kernel kernelNormalize;
	Kernel kernelfindExtreme;
	Kernel kernelMax;
	Kernel kernelInt2Vector;
	Kernel kernelcreateIMG;
	double normaErro;


} *Cnn, TypeCnn;

Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz);

void releaseCnn(Cnn *pc);

Cnn createCnnWithWrapperFile(char *kernelFile, Params p, UINT inx, UINT iny, UINT inz,
                             unsigned long long int devicetype);

Cnn createCnnWithWrapperProgram(const char *kernelprogram, Params p, UINT inx, UINT iny,
                                UINT inz, ULL devicetype);

int CnnCalculeError(Cnn c);

Ponto3d __addLayer(Cnn c);

#define checkSizeFilter(v, tam, pas) ((((v)-(tam))/(pas)) ==((double)(v)-(tam))/((double)(pas)))

int CnnAddConvLayer(Cnn c, char usehost, UINT passo, UINT tamanhoDoFiltro, UINT numeroDeFiltros);

int CnnAddConvNcLayer(Cnn c, char usehost, UINT passox, UINT passoy, UINT largx, UINT largy,
                      UINT filtrox, UINT filtroy,
                      UINT numeroDeFiltros);

int CnnAddPoolLayer(Cnn c, char usehost, UINT passo, UINT tamanhoDoFiltro);

int CnnAddPoolAvLayer(Cnn c, char usehost, UINT passo, UINT tamanhoDoFiltro);

int CnnAddReluLayer(Cnn c, char usehost);

int CnnAddPaddingLayer(Cnn c, char usehost, UINT top, UINT bottom, UINT left, UINT right);

int CnnAddBatchNorm(Cnn c, char usehost, double epsilon);

int CnnAddSoftMax(Cnn c, char usehost);

int CnnAddDropOutLayer(Cnn c, char usehost, double pontoAtivacao, long long int seed);

int CnnAddFullConnectLayer(Cnn c, char usehost, UINT tamanhoDaSaida, int funcaoDeAtivacao);

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