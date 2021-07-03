//
// Created by Henrique on 5/8/2021.
//
#include "CamadaSoftMax.h"
const char *tostringSoftMax(CamadaSoftMax c){
    if(c->super.__string__ != NULL)free(c->super.__string__);
    c->super.__string__ = (char *) calloc(1000, sizeof(char));
    int len = snprintf(c->super.__string__, 1000,
                       "SoftMax  Layer: (%u,%u,%u) -> (%u,%u,%u)\n"

            , c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
                       c->super.saida->x, c->super.saida->y, c->super.saida->z
    );
    len+=1;
    c->super.__string__ = realloc(c->super.__string__, sizeof (char) * len);
    return c->super.__string__;
}
Camada createSoftMax(WrapperCL *cl, cl_command_queue  queue,unsigned int inx, unsigned int iny, unsigned int inz, Tensor entrada, GPU_ERROR *error) {
	if (error->error)return NULL;

	CamadaSoftMax c = (CamadaSoftMax) calloc(1, sizeof(TypecamadaSoftMax));

	__newCamada__((Camada) c, cl, SOFTMAX, entrada, queue, NULL, inx, iny, inz, inx, iny, inz, error);
	c->super.toString = (fch) tostringSoftMax;
	c->super.release = (fv) realeaseSoftMax;
	c->super.ativa = (fv) ativaSoftMax;
	c->super.calc_grads = (fvv) calc_gradsSoftMax;
	c->super.corrige_pesos = (fv) corrige_pesosSoftMax;
	c->super.salvar = (fsl) salvarSoftMax;

	c->soma = newTensor(cl->context,1,1,inz,error);
	c->exponent = newTensor(cl->context,inx,iny,inz,error);

	c->kernelSoftMaxAtiva1 = new_Kernel(cl->program, "SoftMaxativa1", 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT);
	c->kernelSoftMaxAtiva2 = new_Kernel(cl->program, "SoftMaxativa2", 6, K_VOID_P, K_VOID_P, K_VOID_P, K_INT, K_INT, K_INT);
	c->kernelSoftMaxCalcGrads = new_Kernel(cl->program, "softMaxcalcgrad", 4, K_VOID_P, K_VOID_P, K_VOID_P, K_INT);
	return (Camada) c;
}

void realeaseSoftMax(CamadaSoftMax *pc) {
	CamadaSoftMax c = *pc;
	releaseTensor(&c->super.gradsEntrada);
	if (c->super.flag_releaseInput)releaseTensor(&c->super.entrada);
    if(c->super.__string__ != NULL){
        free(c->super.__string__);
    }
	releaseTensor(&c->super.saida);
	releaseTensor(&c->soma);
	releaseTensor(&c->exponent);
	releaseKernel(&c->kernelSoftMaxCalcGrads);
	releaseKernel(&c->kernelSoftMaxAtiva1);
	releaseKernel(&c->kernelSoftMaxAtiva2);
	free(c);
	*pc = NULL;
}

void ativaSoftMax(CamadaSoftMax c) {
	double zero = 0.0;
	clEnqueueFillBuffer(c->super.queue,c->soma->data,&zero,sizeof(double),0,c->soma->bytes,0,NULL,NULL);
	kernel_run_recursive(&c->kernelSoftMaxAtiva1, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z, *c->super.max_works,
	                     &c->super.entrada->data, &c->super.saida->data);
	kernel_run_recursive(&c->kernelSoftMaxAtiva2, c->super.queue,
	                     c->super.saida->x * c->super.saida->y * c->super.saida->z, *c->super.max_works,
	                      &c->exponent->data,&c->soma->data,&c->super.saida->data,&c->super.entrada->x,&c->super.entrada->y);

}

void corrige_pesosSoftMax(CamadaSoftMax c) {}

void calc_gradsSoftMax(CamadaSoftMax c, Tensor GradNext) {
	kernel_run_recursive(&c->kernelSoftMaxCalcGrads, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->y * c->super.entrada->z, *c->super.max_works,
	                     &c->super.gradsEntrada->data, &c->super.entrada->data,
	                     &GradNext->data);

}

void salvarSoftMax(WrapperCL *cl, CamadaSoftMax c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);

}

Camada carregarSoftMax(WrapperCL *cl, FILE *src,cl_command_queue queue, Tensor entrada, Params *params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz;
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	return createSoftMax(cl,queue, inx, iny, inz, entrada, error);
}
