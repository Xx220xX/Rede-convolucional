//
// Created by Henrique on 03/08/2021.
//
#include "../CamadaPadding.h"

const char *getCreateParamsPadding(CamadaPadding c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['Padding',%u,%u,%u,%u]",
	                   c->top, c->bottom, c->left, c->right
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringPadding(CamadaPadding c) {
	if (c->super.__string__ != NULL)free(c->super.__string__);
	c->super.__string__ = (char *) calloc(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "Padding  Layer: (%u,%u,%u) -> (%u,%u,%u)\n"
	                   "\ttop %u, bottom %u, left %d, right %u\n",
	                   c->super.entrada->x, c->super.entrada->y, c->super.entrada->z,
	                   c->super.saida->x, c->super.saida->y, c->super.saida->z,
	                   c->top, c->bottom, c->left, c->right
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

void realeasePadding(CamadaPadding *pc) {
	CamadaPadding c = *pc;
	__releaseCamada__((Camada) c);
	releaseKernel(&c->ativa);
	releaseKernel(&c->calcGrad);
	releaseTensor(&c->super.saida);
	free(c);
	*pc = NULL;
}

int ativaPadding(CamadaPadding c) {
	int erro = kernel_run_recursive(&c->ativa, c->super.queue,
	                                c->super.entrada->x * c->super.entrada->x * c->super.entrada->z,
	                                *c->super.max_works,
	                                &c->super.entrada->data,
	                                &c->super.saida->data,
	                                &c->super.entrada->x,
	                                &c->super.entrada->y,
	                                &c->super.saida->x,
	                                &c->super.saida->y,
	                                &c->top,
	                                &c->left
	);
	return erro;
}

int corrige_pesosPadding(CamadaPadding c) { return 0; }

int calc_gradsPadding(CamadaPadding c, Tensor GradNext) {
	int erro = kernel_run_recursive(&c->calcGrad, c->super.queue,
	                                c->super.entrada->x * c->super.entrada->x * c->super.entrada->z,
	                                *c->super.max_works,
	                                &GradNext->data,
	                                &c->super.gradsEntrada->data,
	                                &c->super.entrada->x,
	                                &c->super.entrada->y,
	                                &c->super.saida->x,
	                                &c->super.saida->y,
	                                &c->top,
	                                &c->left
	);
	return erro;

}

void salvarPadding(WrapperCL *cl, CamadaPadding c, FILE *dst, GPU_ERROR *error) {
	char flag = '#';
	fwrite(&c->super.type, sizeof(char), 1, dst);
	fwrite(&flag, sizeof(char), 1, dst);
	fwrite(&c->super.flag_usehost, sizeof(char), 1, dst);
	fwrite(&c->super.entrada->x, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->y, sizeof(UINT), 1, dst);
	fwrite(&c->super.entrada->z, sizeof(UINT), 1, dst);
	fwrite(&c->top, sizeof(UINT), 1, dst);
	fwrite(&c->bottom, sizeof(UINT), 1, dst);
	fwrite(&c->left, sizeof(UINT), 1, dst);
	fwrite(&c->right, sizeof(UINT), 1, dst);

}

Camada carregarPadding(WrapperCL *cl, FILE *src, cl_command_queue queue,
                       Tensor entrada, Params params, GPU_ERROR *error) {
	if (error->error)return NULL;
	char flag = 0;
	fread(&flag, sizeof(char), 1, src);
	if (flag != '#')
		fread(&flag, sizeof(char), 1, src);
	UINT inx, iny, inz, top, bottom, left, right;
	char flag_usehost = 0;
	fread(&flag_usehost, sizeof(char), 1, src);
	fread(&inx, sizeof(UINT), 1, src);
	fread(&iny, sizeof(UINT), 1, src);
	fread(&inz, sizeof(UINT), 1, src);
	fread(&top, sizeof(UINT), 1, src);
	fread(&bottom, sizeof(UINT), 1, src);
	fread(&left, sizeof(UINT), 1, src);
	fread(&right, sizeof(UINT), 1, src);
	return createPadding(cl, queue, inx, iny, inz, top, bottom, left, right, entrada, flag_usehost,error);
}

Camada createPadding(WrapperCL *cl, QUEUE queue,
                     UINT inx, UINT iny, UINT inz,
                     UINT top, UINT bottom, UINT left, UINT right, Tensor entrada,
                     char usehost, GPU_ERROR *error) {
	if (error->error)return NULL;

	CamadaPadding c = (CamadaPadding) calloc(1, sizeof(TypecamadaPadding));

	__newCamada__((Camada) c, cl, PADDING, entrada, queue,
	              (Params) {0}, inx, iny, inz, inx + top + bottom, iny + left + right,
	              inz,usehost, error);
	c->super.gradsEntrada = newTensor(cl->context, queue, inx, iny, inz, 1, error);
	c->super.toString = (fch) tostringPadding;
	c->super.getCreateParams = (fch) getCreateParamsPadding;
	c->super.release = (fv) realeasePadding;
	c->super.ativa = (fv) ativaPadding;
	c->super.calc_grads = (fvv) calc_gradsPadding;
	c->super.corrige_pesos = (fv) corrige_pesosPadding;
	c->super.salvar = (fsl) salvarPadding;
	c->top = top;
	c->bottom = bottom;
	c->left = left;
	c->right = right;
	c->ativa = new_Kernel(cl->program, error, "paddingfeed", 9,
	                      K_VOID_P, K_VOID_P,
	                      K_INT, K_INT, K_INT,
	                      K_INT, K_INT, K_INT,
	                      K_INT
	);
	c->calcGrad = new_Kernel(cl->program, error, "paddingBack", 9,
	                         K_VOID_P, K_VOID_P,
	                         K_INT, K_INT, K_INT,
	                         K_INT, K_INT, K_INT,
	                         K_INT
	);
	return (Camada) c;
}
