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
/***
 * Armazena os dados de uma rede neural convolucional
 */
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
	Exception error;
} *Cnn, TypeCnn;

/***
 * Cria uma Cnn
 * @param cl interface para API openCL
 * @param p parametros padrao a ser aplicados em todas camadas (somente na inicialização)
 * @param inx entrada da rede dimensão x
 * @param iny entrada da rede dimensão y
 * @param inz entrada da rede dimensão z
 * @return retorna uma Cnn que deve ser liberada com a funcao releaseCnn
 */
Cnn createCnn(WrapperCL *cl, Params p, UINT inx, UINT iny, UINT inz);

/***
 * Libera os recursos alocados pela Cnn
 * @param pc endereço para a cnn
 */
void releaseCnn(Cnn *pc);

/***
 * Cria uma Cnn a partir de um kernel em um arquivo
 * @param kernelFile
 * @param p
 * @param inx
 * @param iny
 * @param inz
 * @param devicetype
 * @return
 */
Cnn createCnnWithWrapperFile(char *kernelFile, Params p, UINT inx, UINT iny, UINT inz,
                             unsigned long long int devicetype);

/***
 * Cria uma Cnn a partir de um kernel em uma string
 * @param kernelprogram
 * @param p
 * @param inx
 * @param iny
 * @param inz
 * @param devicetype
 * @return
 */
Cnn createCnnWithWrapperProgram(const char *kernelprogram, Params p, UINT inx, UINT iny,
                                UINT inz, ULL devicetype);

/***
 * Calcula o erro gerada na saida da rede
 * Deve ser chamado após o CnnLearn
 * O resultado ficará em c->normaErro
 * @param c 
 * @return
 */
int CnnCalculeError(Cnn c);


/***
 *  Adiciona camada conv
 * @param c 
 * @param tensor_flag 
 * @param passo 
 * @param tamanhoDoFiltro 
 * @param numeroDeFiltros 
 * @return 
 */
int CnnAddConvLayer(Cnn c, char tensor_flag, UINT passox,UINT passoy,UINT filtrox,UINT filtroy, UINT numeroDeFiltros);

int CnnAddConvNcLayer(Cnn c, char tensor_flag, UINT passox, UINT passoy, UINT filtrox, UINT filtroy,
                      UINT largx, UINT largy,
                      UINT numeroDeFiltros);

int CnnAddPoolLayer(Cnn c, char tensor_flag, UINT passox, UINT passoy,
                    UINT filtrox, UINT filtroy);

int CnnAddPoolAvLayer(Cnn c, char tensor_flag, UINT passox,UINT pasoy,UINT fx,UINT fy);

int CnnAddReluLayer(Cnn c, char tensor_flag);

int CnnAddPaddingLayer(Cnn c, char tensor_flag, UINT top, UINT bottom, UINT left, UINT right);

int CnnAddBatchNorm(Cnn c, char tensor_flag, double epsilon);

int CnnAddSoftMax(Cnn c, char tensor_flag);

int CnnAddDropOutLayer(Cnn c, char tensor_flag, double pontoAtivacao, long long int seed);

int CnnAddFullConnectLayer(Cnn c, char tensor_flag, UINT tamanhoDaSaida, int funcaoDeAtivacao);

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