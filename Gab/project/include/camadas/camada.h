//
// Created by hslhe on 13/11/2021.
//

#ifndef GAB_CNN_CAMADA_H
#define GAB_CNN_CAMADA_H

#include <gpu/Gpu.h>
#include <gpu/Kernel.h>
#include "kernels/kernels.h"
#include "tensor/tensor.h"
#include "cnn/parametros.h"
#include "cnn/ponto3d.h"
#include "funcoesDeAtivacao.h"
#include "error_list.h"
#include "cwrap_kernels.h"

#include <stdarg.h>

#define CONVOLUCAO_ID         0x1
#define CONVOLUCAOF_ID         0x2
#define CONVOLUCAONC_ID         0x3
#define POOL_ID                     0x4
#define FULLCONNECT_ID             0x5
#define PADDING_ID                 0x6
#define DROPOUT_ID             0x7
#define RELU_ID                 0x8
#define PRELU_ID                 0x9
#define SOFTMAX_ID             0xA
#define BATCHNORM_ID             0xB
#define CONVOLUCAO2D_ID             0xC

/// implementa a maxpooling
#define MAXPOOL                 0x1
/// implementa a minpooling
#define MINPOOL                 0x2
/// implementa a averagepooling
#define AVEPOOL                 0x3

/// Indica que a softmax é a ultima camada .
#define SOFTLAST                 0x1
/// Indica que a softmax vai subtrair o maximo da entrada .
#define SOFTNORM                 0x2
/// Indica que a softmax não é a ultima camada (default) .
#define SOFTNLAST                 0x0
/// Indica que a softmax não vai subtrair o maximo da entrada (default) .
#define SOFTNNORM                 0x0


typedef struct Camada_t {
	/// nome canonico da camada (apenas leitura)
	const char *layer_name;
	/// identificador da camada (apenas leitura)
	const char layer_id;
	/// parametros da camada
	Parametros params;
	/// entrada
	Tensor a;
	/// gradiente de entrada
	Tensor da;
	/// saída
	Tensor s;
	/// fila para utilizar gpu
	void *queue;
	/// numero maximo de threads na gpu
	size_t *maxcompute;
	/// variavel de controle de erro
	Ecx ecx;
	/// tamanho da entrada
	P3d size_in;

	/// faz a propagação na camada, o Tensro de entrada é usado a
	Tensor (*propagation)(void *self, Tensor entrada);

	/// faz a retropropagação, ds deve ser o gradiente da saída
	int (*retroPropagation)(void *self, Tensor ds);

	/// faz a retropropagação em lote, ds deve ser o gradiente da saída
	int (*retroPropagationBatch)(void *self, Tensor ds, size_t batchSize);

	/// faz a retropropagação em lote, ds deve ser o gradiente da saída
	int (*retroPropagationBatchLearn)(void *self);

	/// atualiza o hitlearn
	int (*updateHitLearn)(void *self, size_t iter);

	/// retorna uma string (que deve ser liberada com free_mem) contendo o objeto no formato json
	char *(*json)(void *self, int showTensorValues);

	/// retorna a chamada do construtor que gerou essa camada
	char *(*getGenerate)(void *self);

	/// libera os recursos alocados pela camada
	void (*release)(void *self_p);

	/// salva a camada no arquivo destino (apenas pesos são salvos, os gradientes são descartados)
	int (*save)(void *self, FILE *destino);

	/// print todos os tensores em um arquivo
	int (*fprint)(void *self, FILE *destino, char *format, ...);

	/// retorna o tamanho da saída dessa camada
	P3d (*getOutSize)(void *self);


	// Programa na gpu
	cl_program program;

	// kernels da gpu
	char *kernel;

	// tamanho do kernel
	size_t kernel_len;

} *Camada, Camada_t;

/**
 * Parametros para função aleatoria
 *  Y = X * a + b
 *  @param type: indica o tipo da distribuicao pode ser TENSOR_NORMAL para gaussiana, TENSOR_UNIFORM para uniforme
 *  0 para iniciar por padrão
 *  -1 para nao aleatoriazar o tensor
 */
typedef struct RdParams {
	int type;
	REAL a, b;
} RandomParams, RdParams, Rdp;

#define  RDP(type, ...)((RandomParams){type,## __VA_ARGS__})

void internal_Camada_fprint(void *self, FILE *destino, char *format, va_list v);

void internal_Camada_new(Camada self, Gpu gpu, Queue queue, char layer_id, const char *layer_name, Parametros params, Tensor entrada, P3d dim_in, P3d dim_out, Ecx erro);

void internal_Camada_release(Camada *self);

void internal_compile(Camada self, Gpu gpu);

void internal_putFativacao(char **s, int *len, FAtivacao_t fAtivacao);

char *internal_json(Camada self, int showValues);

void internal_saveCamada(FILE *f, Camada self);

void internal_loadCamada(FILE *f, Parametros *parametros, P3d *size_in, uint32_t *size_element);

void internal_saveTensor(FILE *f, Tensor t);

void internal_loadTensor(FILE *f, Tensor t, uint32_t size_element);

void internal_saveREAL(FILE *f, REAL value);

void internal_loadREAL(FILE *f, REAL *value, uint32_t size_element);

RdParams internal_getDefaultRDP(int is_reluActivation, size_t inputLength, size_t outLength);

int internal_unused(void *a, ...);

int internal_notBatch(Camada self, Tensor ds, size_t batchSize);

int internal_updateHitLearn(Camada self, size_t iter);

#define INTERNAL_DEFAULT_ARGS Gpu gpu, Queue queue, P3d size_in,Tensor entrada, Ecx ecx

#define Execute(kernel, len, ...)if(!self->super.ecx->error)self->super.ecx->setError(self->super.ecx, \
self->kernel->runRecursive(self->kernel, self->super.queue,len,*self->super.maxcompute, ##__VA_ARGS__),"%s:%d %s",__FILE__,__LINE__,__FUNCTION__)

#define ExecuteN(kernel, queue, globals, locals, ecx, ...)if(!ecx->error)ecx->setError(ecx, \
kernel->run(kernel,queue,globals,locals, ##__VA_ARGS__))
#define ReleaseKernel(clkernel)    if(clkernel){clReleaseKernel(clkernel);}clkernel = NULL
#define ReleaseVoid(void_ptr)    if(void_ptr){gab_free(void_ptr);}void_ptr = NULL
#define Release(self)if(self)(self)->release(&(self));(self)=NULL
#define KRN_news(var_dst, fname, arg)var_dst = Kernel_news(gpu->program,fname,arg);CheckKernel(var_dst)
#define KRN_new(var_dst, fname, ...)var_dst = Kernel_new(gpu->program,fname,##__VA_ARGS__);CheckKernel(var_dst)
#define CheckKernel(kernel)if (self->super.ecx->setError(self->super.ecx, kernel->error,"%s:%d %s",__FILE__,__LINE__,__FUNCTION__)){goto methods;}


#define apendTensor(name, t, string, len, tmp, showValues) \
if(self->t)  {                                                     \
tmp = self->t->json(self->t, showValues);\
apendstr(string, len, ",\n"PAD"\""name"\":%s", tmp);\
gab_free(tmp);}\
else apendstr(string, len, ",\n"PAD"\""name"\": null")


#define GEN_LAYERNAME(string, len)apendstr(string, len, "%s (",lname)
#define GEN_P2D(p2d, string, len)apendstr(string, len, "P2D(%zu, %zu)",p2d.x,p2d.y)
#define GEN_P3D(p3d, string, len)apendstr(string, len, "P3D(%zu, %zu, %zu)",p3d.x,p3d.y,p3d.z)
#define GEN_RDP(rdp, string, len)apendstr(string, len, "RDP(%d, %g, %g)",rdp.type,(double)(rdp.a),(double)(rdp.b))
#define GEN_PARAMS(prm, string, len)apendstr(string, len, "Params(%g, %g, %g, %d,%g,%g)",(double)(prm.lr_0),(double)(prm.momento),(double)(prm.decaimento),prm.skipLearn,(double)(prm.a),(double)(prm.b))
#define GENN_P2D(p2d, string, len)apendstr(string, len, "P2D(%zu, %zu), ",p2d.x,p2d.y)
#define GENN_P3D(p3d, string, len)apendstr(string, len, "P3D(%zu, %zu, %zu), ",p3d.x,p3d.y,p3d.z)
#define GENN_RDP(rdp, string, len)apendstr(string, len, "RDP(%d, %g, %g), ",rdp.type,(double)(rdp.a),(double)(rdp.b))
#define GENN_PARAMS(prm, string, len)apendstr(string, len, "Params(%g, %g, %g, %d,%g,%g), ",(double)(prm.lr_0),(double)(prm.momento),(double)(prm.decaimento),prm.skipLearn,(double)(prm.a),(double)(prm.b))

#define GEN_END(string, len)apendstr(string, len, ")")


#define DEFAULT_COD    "#define REAL float\n"\
"#define kMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))\n\n"\
"#define kMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))\n\n"\
"#define kRep4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\\\n"\
"_y_ = total%%ty      ;                                        \\\n"\
"_x_ = (total - _y_)%%(ty*tx)/ty ;                             \\\n"\
"_z_ = (total- _x_*ty - _y_)%%(tx*ty*tz)/(ty*tx)  ;            \\\n"\
"_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);\n\n\n"\
"#define kRap(total, _x_, _y_, _z_, tx, ty)\\\n"\
"_y_ = total %% ty;\\\n"\
"_x_ = ((total - _y_) %% (ty * tx)) / ty;\\\n"\
"_z_ = (k - _x_ * ty - _y_) / (tx * ty);\n\n"\
"#define KRap2D(total, x, y, ty)\\\n"\
"y = total %% ty;\\\n"\
"x = total/ ty;\n\n"\
"#define CORRIGIR_PESOS(peso,gradiente,learnR,decaimento) peso = peso - learnR*(gradiente + decaimento*peso)\n\n\n"
#define COD(fmt, ...)apendstr(self->super.kernel, self->super.kernel_len,fmt,##__VA_ARGS__)

#define IFCOD(cond, fmt, ...)if(cond){apendstr(self->super.kernel, self->super.kernel_len,fmt,##__VA_ARGS__)}
#define ELIFCOD(cond, fmt, ...)else if(cond){apendstr(self->super.kernel, self->super.kernel_len,fmt,##__VA_ARGS__)}
#define ELSECOD(fmt, ...)else{apendstr(self->super.kernel, self->super.kernel_len,fmt,##__VA_ARGS__)}

#define  Super self->super
#define setKernelArg(kernel_v, id, tipo, var)  self->super.ecx->setError(self->super.ecx,clSetKernelArg(kernel_v, id, sizeof(tipo), &var),"%s:%d , %s %s\n",__FILE__,__LINE__,__FUNCTION__,#kernel_v)
#define setKernelArgt(kernel_v, id, var)  self->super.ecx->setError(self->super.ecx,clSetKernelArg(kernel_v, id, sizeof(typeof(var)), &var),"%s:%d , %s %s\n",__FILE__,__LINE__,__FUNCTION__,#kernel_v)
#define runr_kernel(error, kernel_v, iterLen, maxcompute, id_index) \
{\
    int trid = 0;                                            \
                                                        \
    error = setKernelArg(kernel_v, id_index, int, trid);\
    size_t globals = iterLen, locals = 1;                    \
    if(error){goto handle_error;}                                                         \
    if (globals <     (maxcompute)) {\
        locals = globals;\
        error = clEnqueueNDRangeKernel(self->super.queue,kernel_v, 1, NULL, &globals, &locals, 0, NULL, NULL); \
        if(error){goto handle_error;}                                                         \
                                                         \
    } else {\
        size_t resto = globals %     (maxcompute);\
        globals = (globals /     (maxcompute)) *     (maxcompute);\
        locals =     (maxcompute);\
       error =  clEnqueueNDRangeKernel(self->super.queue, kernel_v, 1, NULL, &globals, &locals, 0, NULL, NULL);        \
       if(error){goto handle_error;}                                                                    \
       if (resto) {\
            trid = globals;\
            locals = resto;\
            globals = resto;\
            error = setKernelArg(kernel_v, id_index, int, trid);           \
                if(error){goto handle_error;}                                                         \
            error = clEnqueueNDRangeKernel(self->super.queue, kernel_v, 1, NULL, &globals, &locals, 0, NULL, NULL);     \
            if(error){goto handle_error;}                                                                    \
        }\
    }\
}

#endif //GAB_CNN_CAMADA_H
