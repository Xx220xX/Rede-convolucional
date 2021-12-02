#ifndef CNN_GPU_CNN_H
#define CNN_GPU_CNN_H

#include "config.h"
#include "camadas/Camada.h"
#include "camadas/CamadaConv.h"
#include "camadas/CamadaConvF.h"
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
	size_t len_input;
	size_t len_output;
	QUEUE queue;
	WrapperCL *cl;
	char releaseCL;

	Kernel kernelsub;
	Kernel kerneldiv;
	Kernel kerneldivInt;
	Kernel kernelNormalize;
	Kernel kernelInt2Vector;
	Kernel kernelcreateIMG;
	Kernel kernelputIMG;

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
int CnnCalculeError(Cnn c, REAL *mse);

int CnnCalculeErrorWithOutput(Cnn c, REAL *target, REAL *mse);

int CnnCalculeErrorTWithOutput(Cnn c, Tensor target, REAL *mse);

int CnnGetIndexMax(Cnn c);

int Convolucao(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT numeroDeFiltros, RandomParam randomParam);

int ConvolucaoF(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT numeroDeFiltros, int funcAtivacao, RandomParam randomParam);

int ConvolucaoNcausal(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy, UINT largx, UINT largy, UINT numeroDeFiltros, RandomParam randomParam);

int Pooling(Cnn c, UINT passox, UINT passoy, UINT filtrox, UINT filtroy);

int PoolingAv(Cnn c, UINT passox, UINT pasoy, UINT fx, UINT fy);

int Relu(Cnn c, REAL lessoh, REAL greateroh);

int PRelu(Cnn c, RandomParam randomParam);

int Padding(Cnn c, UINT top, UINT bottom, UINT left, UINT right);

int BatchNorm(Cnn c, REAL epsilon, RandomParam randomParamY, RandomParam randomParamB);

int SoftMax(Cnn c);

int Dropout(Cnn c, REAL pontoAtivacao, long long int seed);

int FullConnect(Cnn c, UINT tamanhoDaSaida, int funcaoDeAtivacao, RandomParam randomParam);

int CnnCall(Cnn c, REAL *input);

int CnnCallT(Cnn c, Tensor input);

int CnnLearn(Cnn c, REAL *target);

int CnnLearnT(Cnn c, Tensor target);

void CnnInitLuaVm(Cnn c);

int CnnLuaConsole(Cnn c);

void LuaputHelpFunctionArgs(void (*myf)());

int CnnLuaLoadString(Cnn c, const char *lua_program);

int CnnLuaLoadFile(Cnn c, const char *file_name);

void cnnSave(Cnn c, FILE *dst);


int cnnCarregar(Cnn c, FILE *src);

void normalizeGPU(Cnn c, REAL *input, REAL *output, int len, REAL maximo, REAL minimo);

void normalizeGPUSpaceKnow(Cnn c, REAL *input, REAL *output, int len, REAL input_maximo,
						   REAL input_minimo,
						   REAL maximo, REAL minimo);

void printCnn(Cnn c);

char *salveCnnOutAsPPMGPU(Cnn c, size_t *h_r, size_t *w_r);

char *salveCnnOutAsPPMGPUR(Cnn c, size_t height, size_t width);

const char *getVersion();

void showVersion();


#endif