//
// Created by Henrique on 5/8/2021.
//
#include "../CamadaRelu.h"
#include "../Camada.h"

const char *getCreateParamsRelu(CamadaRelu c){
    if(c->super.__string__ != NULL)free(c->super.__string__);
    c->super.__string__ = (char *) calloc(1000, sizeof(char));
    int len = snprintf(c->super.__string__, 1000,
                       "['Relu']"
    );
    len+=1;
    c->super.__string__ = realloc(c->super.__string__, sizeof (char) * len);
    return c->super.__string__;
}
const char *tostringRelu(CamadaRelu c){
    if(c->super.__string__ != NULL)free(c->super.__string__);
    c->super.__string__ = (char *) calloc(1000, sizeof(char));
    int len = snprintf(c->super.__string__, 1000,
                       "Relu  Layer: (%u,%u,%u) -> (%u,%u,%u)\n"

                       , c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
                       c->super.saida->x, c->super.saida->y, c->super.saida->z
    );
    len+=1;
    c->super.__string__ = realloc(c->super.__string__, sizeof (char) * len);
    return c->super.__string__;
}
Camada createRelu(WrapperCL *cl, cl_command_queue  queue,unsigned int inx, unsigned int iny,
				  unsigned int inz, Tensor entrada, GPU_ERROR *error) {
	if (error->error)return NULL;

	CamadaRelu c = (CamadaRelu) calloc(1, sizeof(TypecamadaRelu));

	__newCamada__((Camada) c, cl, RELU, entrada, queue, (Params){0}, inx, iny, inz, inx, iny, inz, error);
	c->super.toString = (fch) tostringRelu;
	c->super.getCreateParams = (fch) getCreateParamsRelu;
	c->super.release = (fv) realeaseRelu;
	c->super.ativa = (fv) ativaRelu;
	c->super.calc_grads = (fvv) calc_gradsRelu;
	c->super.corrige_pesos = (fv) corrige_pesosRelu;
	c->super.salvar = (fsl) salvarRelu;

	c->kernelReluAtiva = new_Kernel(cl->program,error, "reluativa", 3, K_VOID_P, K_VOID_P, K_INT);
	c->kernelReluCalcGrads = new_Kernel(cl->program,error, "relucalcgrad", 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	return (Camada) c;
}

void realeaseRelu(CamadaRelu *pc) {
	CamadaRelu c = *pc;
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
    if(c->super.__string__ != NULL){
        free(c->super.__string__);
    }
	releaseTensor(&c->super.saida);
	releaseKernel(&c->kernelReluCalcGrads);
	releaseKernel(&c->kernelReluAtiva);
	free(c);
	*pc = NULL;
}

int ativaRelu(CamadaRelu c) {
	int erro = kernel_run_recursive(&c->kernelReluAtiva, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z, *c->super.max_works,
	                     &c->super.entrada->data, &c->super.saida->data);

	return erro;
}

int corrige_pesosRelu(CamadaRelu c) {return 0;}

int calc_gradsRelu(CamadaRelu c, Tensor GradNext) {
	int erro = kernel_run_recursive(&c->kernelReluCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z, *c->super.max_works,
	                     &c->super.gradsEntrada->data, &c->super.entrada->data,
	                     &GradNext->data);
	return erro;

}

void salvarRelu(WrapperCL *cl, CamadaRelu c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);

}

Camada carregarRelu(WrapperCL *cl, FILE *src,cl_command_queue queue, Tensor entrada, Params params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createRelu(cl, queue,inx, iny, inz, entrada, error);
}

void CamadaSetLearn(Camada c, char learn) {
	c->flag_notlearn = !learn;
}

void CamadaSetParams(Camada c, double hitlearn, double momento, double decaimento) {
	c->parametros.hitLearn = hitlearn;
	c->parametros.momento = momento;
	c->parametros.decaimentoDePeso = decaimento;
}
