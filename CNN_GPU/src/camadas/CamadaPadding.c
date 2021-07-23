//
// Created by Henrique on 03/08/2021.
//
#include "camadas/CamadaPadding.h"
#if  defined(DISABLE_KERNELS_INSIDE_DRIVE)
#include "../../../kernels/camadas/utils.h"
#include "../../../kernels/camadas/padding.h"
#endif
const char *getCreateParamsPadding(CamadaPadding c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
	int len = snprintf(c->super.__string__, 1000,
	                   "['Padding',%u,%u,%u,%u]",
	                   c->top, c->bottom, c->left, c->right
	);
	len += 1;
	c->super.__string__ = realloc(c->super.__string__, sizeof(char) * len);
	return c->super.__string__;
}

const char *tostringPadding(CamadaPadding c) {
	if (c->super.__string__ != NULL)free_mem(c->super.__string__);
	c->super.__string__ = (char *) alloc_mem(1000, sizeof(char));
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
	free_mem(c);
	*pc = NULL;
}

int ativaPadding(CamadaPadding c) {
	int erro = 0;
	kernel_run_recursive(erro, c->ativa, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->x * c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG c->super.entrada->data,
	                     K_ARG c->super.saida->data,
	                     K_ARG c->super.entrada->x,
	                     K_ARG c->super.entrada->y,
	                     K_ARG c->super.saida->x,
	                     K_ARG c->super.saida->y,
	                     K_ARG c->top,
	                     K_ARG c->left
	);
	return erro;
}

int corrige_pesosPadding(CamadaPadding c) { return 0; }

int calc_gradsPadding(CamadaPadding c, Tensor GradNext) {
	if (!c->super.gradsEntrada)return 0;
	int erro = 0;
	kernel_run_recursive(erro, c->calcGrad, c->super.queue,
	                     c->super.entrada->x * c->super.entrada->x * c->super.entrada->z,
	                     *c->super.max_works,
	                     K_ARG GradNext->data,
	                     K_ARG c->super.gradsEntrada->data,
	                     K_ARG c->super.entrada->x,
	                     K_ARG c->super.entrada->y,
	                     K_ARG c->super.saida->x,
	                     K_ARG c->super.saida->y,
	                     K_ARG c->top,
	                     K_ARG c->left
	);
	return erro;

}

void salvarPadding(WrapperCL *cl, CamadaPadding c, FILE *dst, Exception *error) {
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
                       Tensor entrada, Params params, Exception *error) {
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
	return createPadding(cl, queue, inx, iny, inz, top, bottom, left, right, entrada, flag_usehost, error);
}

Camada createPadding(WrapperCL *cl, QUEUE queue,
                     UINT inx, UINT iny, UINT inz,
                     UINT top, UINT bottom, UINT left, UINT right, Tensor entrada,
                     char usehost, Exception *error) {
	if (error->error)return NULL;

	CamadaPadding c = (CamadaPadding) alloc_mem(1, sizeof(TypecamadaPadding));

	__newCamada__((Camada) c, cl, PADDING, entrada, queue,
	              (Params) {0}, inx, iny, inz, inx + top + bottom, iny + left + right,
	              inz, usehost, error);
	error->error = TensorFill(queue, c->super.saida, 0);
	c->super.toString = (cfv) tostringPadding;
	c->super.getCreateParams = (cfv) getCreateParamsPadding;
	c->super.release = (fv) realeasePadding;
	c->super.ativa = (fv) ativaPadding;
	c->super.calc_grads = (f2v) calc_gradsPadding;
	c->super.corrige_pesos = (fv) corrige_pesosPadding;
	c->super.salvar = (f4v) salvarPadding;
	c->top = top;
	c->bottom = bottom;
	c->left = left;
	c->right = right;
	c->ativa = new_Kernel(cl->program, error, paddingfeed, 9,
	                      K_VOID_P, K_VOID_P,
	                      K_INT, K_INT, K_INT,
	                      K_INT, K_INT, K_INT,
	                      K_INT
	);
	c->calcGrad = new_Kernel(cl->program, error, paddingBack, 9,
	                         K_VOID_P, K_VOID_P,
	                         K_INT, K_INT, K_INT,
	                         K_INT, K_INT, K_INT,
	                         K_INT
	);
	if (error->error) {
		getClError(error->error, error->msg, EXCEPTION_MAX_MSG_SIZE);
	}
	return (Camada) c;
}
