//
// Created by Xx220xX on 26/10/2020.
//

#ifndef CNN_GPU_CAMADAFULLCONNECT_H
#define CNN_GPU_CAMADAFULLCONNECT_H

#include "string.h"
#include"Camada.h"
#include"Tensor.h"
#include <stdlib.h>
#include <float.h>

#include"funcoesDeAtivacao.h"

typedef unsigned int UINT;

typedef struct {
	Typecamada super;
	Tensor input;
	Tensor pesos;
	Tensor grad;
	Tensor oldgrad;
	// funcao de ativacao e sua derivada
	int fa, dfa;
	Kernel kernelfullfeed;
	Kernel kernelfullfixWeight;
	Kernel kernelfullcalcgrad1;
	Kernel kernelfullcalcgrad2;
} *CamadaFullConnect, Typecamadafullconnect;

void releaseFullConnect(CamadaFullConnect *pc);


void corrigePesosFullConnect(CamadaFullConnect c);

void ativaFullConnect(CamadaFullConnect c);

void calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext);

int fullRandomize(CamadaFullConnect c, WrapperCL *cl, GPU_ERROR *error);

void salvarFullConnect(WrapperCL *cl, CamadaFullConnect c, FILE *dst, GPU_ERROR *error);

Camada createFullConnect(WrapperCL *cl, UINT inx, UINT iny, UINT inz, UINT tamanhoSaida, Tensor entrada, Params *params,
                         int funcaoDeAtivacao, int randomize, GPU_ERROR *error) {
	if (error->error)return NULL;
	CamadaFullConnect c = (CamadaFullConnect) calloc(1, sizeof(Typecamadafullconnect));
	cl_context context = cl->context;
	c->super.gradsEntrada = newTensor(context, inx, iny, inz, error);
	c->super.parametros = params;
	if (!entrada) {
		c->super.entrada = newTensor(context, inx, iny, inz, error);
		c->super.flag_releaseInput = 1;
	} else {
		c->super.entrada = entrada;
	}

	c->super.saida = newTensor(context, tamanhoSaida, 1, 1, error);
	c->input = newTensor(context, tamanhoSaida, 1, 1, error);
	c->grad = newTensor(context, tamanhoSaida, 1, 1, error);
	c->oldgrad = newTensor(context, tamanhoSaida, 1, 1, error);

	c->pesos = newTensor(context, inx * iny * inz, tamanhoSaida, 1, error);

	if (randomize) {
		fullRandomize(c, cl, error);
	}
	c->super.release = (fv) releaseFullConnect;
	c->super.ativa = (fv) ativaFullConnect;
	c->super.calc_grads = (fvv) calc_gradsFullConnect;
	c->super.corrige_pesos = (fv) corrigePesosFullConnect;
	c->super.type = FULLCONNECT;
	c->fa = funcaoDeAtivacao;
	c->dfa = funcaoDeAtivacao | FLAGDIF;
	c->super.salvar = (fsl) salvarFullConnect;

	c->kernelfullfeed = new_Kernel(cl->program, "fullfeed", 11, VOID_P, VOID_P, VOID_P, VOID_P,
	                               INT, INT, INT, INT, INT, INT, INT);
//    fullfixweight(gdouble *entrada, gdouble *pesos, gdouble *grad, gdouble *oldgrad,
//    double hitlearn, double decaimentoDePeso, double momento,
//    int inx, int iny, int inz, int pesosx, int pesosy, int k0) {
	c->kernelfullfixWeight = new_Kernel(cl->program, "fullfixweight", 13, VOID_P, VOID_P, VOID_P, VOID_P,
	                                    DOUBLE, DOUBLE, DOUBLE,
	                                    INT, INT, INT, INT, INT, INT);
	c->kernelfullcalcgrad1 = new_Kernel(cl->program, "fullcalcgrads1", 5, VOID_P, VOID_P, VOID_P, INT, INT);
	c->kernelfullcalcgrad2 = new_Kernel(cl->program, "fullcalcgrads2", 6, VOID_P, VOID_P, VOID_P, INT, INT, INT);

	return (Camada) c;
}

int fullRandomize(CamadaFullConnect c, WrapperCL *cl, GPU_ERROR *error) {
	unsigned int inx = c->super.entrada->x;
	unsigned int iny = c->super.entrada->y;
	unsigned int inz = c->super.entrada->z;
	unsigned int tamanhoSaida = c->super.saida->x;
//    unsigned int valmax = inx * iny * inz;
	unsigned int valmax = sqrt(inx * iny * inz)+1;

	double *data = callocdouble(inx * iny * inz * tamanhoSaida);
	for (int i = 0; i < tamanhoSaida; ++i) {
		for (int j = 0; j < valmax; ++j) {
			data[TensorMap(c->pesos, j, i, 0)] = 2.19722 / (valmax) * RANDOM_BILATERAL();
		}
	}
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
	error->error = clEnqueueWriteBuffer(queue, c->pesos->data, CL_TRUE, 0, c->pesos->bytes, data, 0, NULL, NULL);
	if (error->error) {
		snprintf(error->msg, 255, "nao foi possivel copiar dados\n");
		free(data);
		clReleaseCommandQueue(queue);
		return error->error;

	}
	clFinish(queue);
	clReleaseCommandQueue(queue);
	free(data);
}

void releaseFullConnect(CamadaFullConnect *pc) {
	CamadaFullConnect c = *pc;
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
	releaseTensor(&c->super.gradsEntrada);
	releaseTensor(&c->pesos);
	releaseTensor(&c->grad);
	releaseTensor(&c->oldgrad);
	releaseTensor(&c->super.saida);
	releaseTensor(&c->input);
	Kernel_release(&c->kernelfullfixWeight);
	Kernel_release(&c->kernelfullfeed);
	Kernel_release(&c->kernelfullcalcgrad1);
	Kernel_release(&c->kernelfullcalcgrad2);
	free(c);
	*pc = 0;

}

void ativaFullConnect(CamadaFullConnect c) {
	int error = 0, id = 0;
	size_t global, local, resto;
	LOG_CNN_KERNELCALL("ativa fullConnect: ativafullConnect")
	call_kernel(c->super.saida->x,
	            Kernel_putArgs(&c->kernelfullfeed, 11, &c->super.entrada->data, &c->pesos->data, &c->input->data,
	                           &c->super.saida->data, &c->fa, &c->super.entrada->x, &c->super.entrada->y,
	                           &c->super.entrada->z, &c->pesos->x, &c->pesos->y, &id);
			            error = clEnqueueNDRangeKernel(c->super.queue, c->kernelfullfeed.kernel, 1, NULL, &global,
			                                           &local, 0, NULL, NULL);
			            PERRW(error, "falha ao chamar kernel ativafullConnect")
	);

}

void corrigePesosFullConnect(CamadaFullConnect c) {
	int error = 0, id = 0;
	size_t global, local, resto;
	LOG_CNN_KERNELCALL("corrige fullConnect: fixfullConnect")

	call_kernel(c->super.saida->x,
	            Kernel_putArgs(&c->kernelfullfixWeight, 13, &c->super.entrada->data, &c->pesos->data, &c->grad->data,
	                           &c->oldgrad->data,
	                           &c->super.parametros->hitLearn, &c->super.parametros->decaimentoDePeso,
	                           &c->super.parametros->momento,
	                           &c->super.entrada->x, &c->super.entrada->y, &c->super.entrada->z, &c->pesos->x,
	                           &c->pesos->y, &id);
			            error = clEnqueueNDRangeKernel(c->super.queue, c->kernelfullfixWeight.kernel, 1, NULL, &global,
			                                           &local, 0, NULL, NULL);
			            PERRW(error, "falha ao chamar kernel fixfullConnect")
	);

}

void calc_gradsFullConnect(CamadaFullConnect c, Tensor GradNext) {
	int error = 0, id = 0;
	size_t global, local, resto;
	LOG_CNN_KERNELCALL("calcgrad fullConnect: calcgrad")
	call_kernel(c->super.saida->x,
	            Kernel_putArgs(&c->kernelfullcalcgrad1, 5, &c->grad->data, &GradNext->data, &c->input->data, &c->dfa,
	                           &id);
			            error = clEnqueueNDRangeKernel(c->super.queue, c->kernelfullcalcgrad1.kernel, 1, NULL, &global,
			                                           &local, 0, NULL, NULL);
			            PERRW(error, "falha ao chamar kernel fullcalgrads1")
	);
	LOG_CNN_KERNELCALL("calcgrad2 fullConnect: calcgrad")
	call_kernel(c->super.entrada->x * c->super.entrada->y * c->super.entrada->z,
	            Kernel_putArgs(&c->kernelfullcalcgrad2, 6, &c->grad->data, &c->super.gradsEntrada->data,
	                           &c->pesos->data, &c->pesos->x, &c->pesos->y, &id);
			            error = clEnqueueNDRangeKernel(c->super.queue, c->kernelfullcalcgrad2.kernel, 1, NULL, &global,
			                                           &local, 0, NULL, NULL);
			            PERRW(error, "falha ao chamar kernel fullcalgrads2")
	);

}

void salvarFullConnect(WrapperCL *cl, CamadaFullConnect c, FILE *dst, GPU_ERROR *error) {
	LOG_CNN_SALVE_LAYERS("Salvando FullConnect")
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->fa, sizeof(int), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->super.saida->x, sizeof(UINT), 1, dst);
	double *data = callocdouble(c->pesos->x * c->pesos->y * c->pesos->z);
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
	clEnqueueReadBuffer(queue, c->pesos->data, CL_TRUE, 0, c->pesos->bytes, data, 0, NULL, NULL);
	fwrite(data, 1, c->pesos->bytes, dst);
	clFinish(queue);
	clReleaseCommandQueue(queue);
	LOG_CNN_SALVE_LAYERS("salvou com erro %d: %s", error->error, error->msg)
}

Camada carregarFullConnect(WrapperCL *cl, FILE *src, Tensor entrada, Params *params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz, tamanhoSaida;
	int fa;
	fread(&fa, sizeof(int), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&tamanhoSaida, sizeof(UINT), 1, src);
	CamadaFullConnect c = (CamadaFullConnect) createFullConnect(cl, inx, iny, inz, tamanhoSaida, entrada, params, fa, 0,
	                                                            error);
	double *data = callocdouble(c->pesos->x * c->pesos->y * c->pesos->z);
	cl_command_queue queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &error->error);
	fread(data, 1, c->pesos->bytes, src);
	clEnqueueWriteBuffer(queue, c->pesos->data, CL_TRUE, 0, c->pesos->bytes, data, 0, NULL, NULL);
	clFinish(queue);
	clReleaseCommandQueue(queue);
	return (Camada) c;
}

#endif //CNN_GPU_CAMADAFULLCONNECT_H
