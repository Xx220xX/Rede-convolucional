//
// Created by Henrique on 19/11/2021.
//
/***
 * Camada fullconnect aplica as equações de uma rede profunda.
 * Condensando todas as dimensões da entrada em um unico vetor coluna  de dimensão(x*y*z,1,1)
 * A matriz de pesos (w) é da dimensão (n,x*y*z,1), onde n é o numero de neuronios de saída.
 * A matriz de bias (b) é da dimensão (n,1,1), onde n é o numero de neuronios de saída.
 *
 * Nesta implementação a matriz de saída é transposta, possuindo dimensão s' = (1,n,1)
 * @Propagation
 * A propagação é feita por
 * z = w*a + b
 * s = f(z)
 *
 * @Retropropagação
 * ds -> gradiente da saída
 * dz = f'(z)*ds
 * dw = dz*a
 * db = dz
 * da = w(T)*dz
 *
 */
#include <math.h>
#include "camadas/CamadaFullConnect.h"

static const char *lname = "FullConnect";

void CamadaFullConnect_release(CamadaFullConnect *self) {
	internal_Camada_release((Camada *) self);
	Release((*self)->w);
	Release((*self)->dw);
	Release((*self)->b);
	Release((*self)->db);
	Release((*self)->z);
	Release((*self)->dz);
	Release((*self)->expoente);
	#if COMPATIBLE_KERAS == 1
	if ((*self)->flatten) {
		Release((*self)->super.a);
		Release((*self)->daf);
	}
	#endif
	ReleaseKernel((*self)->feed);
	ReleaseKernel((*self)->calc_exp);
	ReleaseKernel((*self)->feed);
	ReleaseKernel((*self)->ativa);
	ReleaseKernel((*self)->calc_exp);
	ReleaseKernel((*self)->calc_dzdb);
	ReleaseKernel((*self)->calc_dzdb_batch);
	ReleaseKernel((*self)->corrige_peso);
	ReleaseKernel((*self)->calc_da);
	ReleaseKernel((*self)->calc_dw);
	ReleaseKernel((*self)->calc_dw_batch);

	ReleaseVoid((*self)->values);
	gab_free(*self);
}

void flatten(Tensor A, Tensor AF) {
	REAL a[A->length];
	REAL af[AF->length];
	A->getvalues(A, a);
	int aX = A->x;
	int aY = A->y;
	int aZ = A->z;
	for (int m = 0; m < A->length; ++m) {
		int x, y, z;
		kRap(m, y, z, x, aY, aZ)
		af[m] = a[kMap(x, y, z, aX, aY)];
	}
	AF->setvalues(AF, af);
}

void reflatten(Tensor dA, Tensor dAF) {
	REAL da[dA->length];
	REAL daf[dAF->length];
	dAF->getvalues(dAF, daf);
	int aX = dA->x;
	int aY = dA->y;
	int aZ = dA->z;
	for (int m = 0; m < dA->length; ++m) {
		int x, y, z;
		kRap(m, y, z, x, aY, aZ)
		da[kMap(x, y, z, aX, aY)] = daf[m];
	}
	dA->setvalues(dA, da);
}


void CamadaFullConnect_calc_da(CamadaFullConnect self) {
	#if COMPATIBLE_KERAS == 1
	if (self->flatten) {
		setKernelArg(self->calc_da, 0, void *, self->daf->data);
		setKernelArg(self->calc_da, 1, void *, self->w->data);
		setKernelArg(self->calc_da, 2, void *, self->dz->data);
		runr_kernel(Super.ecx->error, self->calc_da, Super.da->length, *Super.maxcompute, 3);
		reflatten(Super.da, self->daf);
		return;
	}
	#endif
	setKernelArg(self->calc_da, 0, void *, Super.da->data);
	setKernelArg(self->calc_da, 1, void *, self->w->data);
	setKernelArg(self->calc_da, 2, void *, self->dz->data);
	runr_kernel(Super.ecx->error, self->calc_da, Super.da->length, *Super.maxcompute, 3);
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return;
}

Tensor CamadaFullConnect_propagation(CamadaFullConnect self, Tensor a) {
	#if COMPATIBLE_KERAS == 1
	if (self->flatten) {
		flatten(a, Super.a);
	} else {
		Super.a = a;
	}
	#else
	Super.a = a;
	#endif
	setKernelArg(self->feed, 0, void *, Super.a->data);
	setKernelArg(self->feed, 1, void *, self->w->data);
	setKernelArg(self->feed, 2, void *, self->b->data);
	setKernelArg(self->feed, 3, void *, self->z->data);
	setKernelArg(self->feed, 5, void *, Super.s->data);
	runr_kernel(Super.ecx->error, self->feed, Super.s->length, *Super.maxcompute, 4)
	return Super.s;
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	fprintf(stderr, "Error %d\n", Super.ecx->error);
	return NULL;
}

Tensor CamadaFullConnect_propagation_softmax(CamadaFullConnect self, Tensor a) {
	#if COMPATIBLE_KERAS == 1
	if (self->flatten) {
		flatten(a, Super.a);
	} else {
		Super.a = a;
	}
	#else
	Super.a = a;
	#endif
	setKernelArg(self->feed, 0, void *, Super.a->data);
	setKernelArg(self->feed, 1, void *, self->w->data);
	setKernelArg(self->feed, 2, void *, self->b->data);
	setKernelArg(self->feed, 3, void *, self->z->data);
	runr_kernel(Super.ecx->error, self->feed, Super.s->length, *Super.maxcompute, 4)
	ECX_IF_FAILED(Super.ecx, handle_error)

	// calcular o maximo;
	self->values = self->z->getvalues(self->z, self->values);
	self->maximo = self->values[0];
	ECX_TRY(Super.ecx, isinf(self->maximo), handle_error, GAB_INFINITO_ENCONTRADO, "Infinito encontrado");
	ECX_TRY(Super.ecx, isnan(self->maximo), handle_error, GAB_NAN_ENCONTRADO, "NAN encontrado");

	for (int i = 1; i < self->z->length; ++i) {
		if (self->values[i] > self->maximo) {
			self->maximo = self->values[i];
		}
	}

	setKernelArg(self->calc_exp, 0, void *, self->expoente->data);
	setKernelArg(self->calc_exp, 1, void *, self->z->data);
	setKernelArg(self->calc_exp, 2, REAL, self->maximo);
	runr_kernel(Super.ecx->error, self->calc_exp, self->z->length, *Super.maxcompute, 3);
	ECX_IF_FAILED(Super.ecx, handle_error)
	self->values = self->expoente->getvalues(self->expoente, self->values);
	ECX_IF_FAILED(Super.ecx, handle_error)
	self->soma = 0;
	for (int i = 0; i < self->z->length; ++i) {
		self->soma += self->values[i];
	}
	setKernelArg(self->ativa, 0, void *, self->expoente->data);
	setKernelArg(self->ativa, 1, void *, Super.s->data);
	setKernelArg(self->ativa, 2, REAL, self->soma);
	runr_kernel(Super.ecx->error, self->ativa, Super.s->length, *Super.maxcompute, 3);
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.s;
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	fprintf(stderr, "Error %d\n", Super.ecx->error);
	return NULL;
}

int CamadaFullConnect_backpropagation(CamadaFullConnect self, Tensor ds) {
	// calcula dz e db e arruma b
	setKernelArg(self->calc_dzdb, 0, void *, self->b->data);
	setKernelArg(self->calc_dzdb, 1, void *, self->db->data);
	setKernelArg(self->calc_dzdb, 2, void *, self->dz->data);
	setKernelArg(self->calc_dzdb, 3, void *, ds->data);
	setKernelArg(self->calc_dzdb, 4, void *, self->z->data);
	setKernelArg(self->calc_dzdb, 5, void *, Super.s->data);
	setKernelArg(self->calc_dzdb, 6, REAL, Super.params.hitlearn);
	runr_kernel(Super.ecx->error, self->calc_dzdb, self->z->length, *Super.maxcompute, 7);
	ECX_IF_FAILED(Super.ecx, handle_error)
	// calcula da
	if (Super.da) {
		CamadaFullConnect_calc_da(self);
		ECX_IF_FAILED(Super.ecx, handle_error)
	}
	// calcula dw e arruma
	setKernelArg(self->calc_dw, 0, void *, self->dw->data);
	setKernelArg(self->calc_dw, 1, void *, self->dz->data);
	setKernelArg(self->calc_dw, 2, void *, Super.a->data);
	setKernelArg(self->calc_dw, 3, void *, self->w->data);
	setKernelArg(self->calc_dw, 4, REAL, Super.params.hitlearn);
	runr_kernel(Super.ecx->error, self->calc_dw, self->dw->length, *Super.maxcompute, 5);
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;
}

int CamadaFullConnect_backpropagationBatch(CamadaFullConnect self, Tensor ds, size_t batchSize) {
	// calcula dz e db
	setKernelArg(self->calc_dzdb_batch, 0, void *, self->db->data);
	setKernelArg(self->calc_dzdb_batch, 1, void *, self->dz->data);
	setKernelArg(self->calc_dzdb_batch, 2, void *, ds->data);
	setKernelArg(self->calc_dzdb_batch, 3, void *, self->z->data);
	setKernelArg(self->calc_dzdb_batch, 4, void *, Super.s->data);
	setKernelArg(self->calc_dzdb_batch, 5, cl_long, batchSize);
	runr_kernel(Super.ecx->error, self->calc_dzdb_batch, self->z->length, *Super.maxcompute, 6);
	ECX_IF_FAILED(Super.ecx, handle_error)

	// calcula da
	if (Super.da) {
		if (Super.da) {
			CamadaFullConnect_calc_da(self);
			ECX_IF_FAILED(Super.ecx, handle_error)
		}
	}
	// calcula dw
	setKernelArg(self->calc_dw_batch, 0, void *, self->dw->data);
	setKernelArg(self->calc_dw_batch, 1, void *, self->dz->data);
	setKernelArg(self->calc_dw_batch, 2, void *, Super.a->data);
	setKernelArg(self->calc_dw_batch, 3, cl_long, batchSize);
	runr_kernel(Super.ecx->error, self->calc_dw_batch, self->dw->length, *Super.maxcompute, 4);

	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;

}

int CamadaFullConnect_learnBatch(CamadaFullConnect self) {
	setKernelArg(self->corrige_peso, 0, void *, self->w->data);
	setKernelArg(self->corrige_peso, 1, void *, self->dw->data);
	setKernelArg(self->corrige_peso, 2, REAL, Super.params.hitlearn);
	runr_kernel(Super.ecx->error, self->corrige_peso, self->dw->length, *Super.maxcompute, 3);
	ECX_IF_FAILED(Super.ecx, handle_error)

	setKernelArg(self->corrige_peso, 0, void *, self->b->data);
	setKernelArg(self->corrige_peso, 1, void *, self->db->data);
	setKernelArg(self->corrige_peso, 2, REAL, Super.params.hitlearn);
	runr_kernel(Super.ecx->error, self->corrige_peso, self->db->length, *Super.maxcompute, 3);

	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;
}

char *CamadaFullConnect_json(CamadaFullConnect self, int showValues) {
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"funcaoAtivacao\":%d", tmp, self->fa.id);
	gab_free(tmp);

	apendTensor("z", z, string, len, tmp, showValues);
	apendTensor("dz", dz, string, len, tmp, showValues);
	apendTensor("w", w, string, len, tmp, showValues);
	apendTensor("dw", dw, string, len, tmp, showValues);

	apendstr(string, len, "\n}");
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return string;
}

char *CamadaFullConnect_getGenerate(CamadaFullConnect self) {
	char *string = NULL;
	int len = 0;
	apendstr(string, len, "%s (%zu, ", lname, self->w->x);
	internal_putFativacao(&string, &len, self->fa.mask);
	apendstr(string, len, ", Params(%g, %g, %g, %d), RDP(%d, %g, %g), RDP(%d, %g, %g))", (double) Super.params.hitlearn, (double) Super.params.momento, (double) Super.params.decaimento, Super.params.skipLearn, self->rdp_pesos.type, self->rdp_pesos.a, self->rdp_pesos.b, self->rdp_bias.type, self->rdp_bias.a, self->rdp_bias.b);
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return string;
}


int CamadaFullConnect_save(CamadaFullConnect self, FILE *f) {
	if (Super.ecx->error) {
		goto end;
	}

	internal_saveCamada(f, (Camada) self);
	fwrite(&self->fa, 1, sizeof(uint32_t), f);
	fwrite(&Super.s->y, 1, sizeof(size_t), f);
	internal_saveTensor(f, self->w);
	internal_saveTensor(f, self->b);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;
}

Camada CamadaFullConnect_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {

	Parametros parametros;
	P3d size_in;
	uint32_t size_element;

	uint32_t fa;
	size_t neuronios;

	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&fa, sizeof(uint32_t), 1, f);
	fread(&neuronios, sizeof(size_t), 1, f);
	CamadaFullConnect self = (CamadaFullConnect) CamadaFullConnect_new(gpu, queue, size_in, neuronios, entrada, parametros, fa, ecx, RDP(-1), RDP(-1));
	internal_loadTensor(f, self->w, size_element);
	internal_loadTensor(f, self->b, size_element);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return (Camada) self;
}

int CamadaFullConnect_fprintf(CamadaFullConnect self, FILE *destino, char *format, ...) {
	va_list v;
	va_start(v, format);
	internal_Camada_fprint(self, destino, format, v);
	fprintf(destino, "w -> ");
	self->w->fprint(self->w, destino);
	fprintf(destino, "dw -> ");
	self->dw->fprint(self->dw, destino);
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return 0;
}

Camada CamadaFullConnect_new(Gpu gpu, Queue queue, P3d size_in, size_t tamanhoSaida, Tensor entrada, Parametros params, FAtivacao_t funcaoDeAtivacao, Ecx ecx, RandomParams rdp_pesos, RandomParams rdp_bias) {

	CamadaFullConnect self = gab_alloc(1, sizeof(CamadaFullConnect_t));
	P3d size_out = {1, tamanhoSaida, 1};
	internal_Camada_new((Camada) self, gpu, queue, FULLCONNECT_ID, lname, params, entrada, size_in, size_out, ecx);
	#if COMPATIBLE_KERAS == 1
	if (size_in.x != 1 || size_in.z != 1) {
		self->flatten = 1;
		Super.a = Tensor_new(1, size_in.x * size_in.y * size_in.z, 1, 1, ecx, 0, gpu->context, queue);
		self->daf = Tensor_new(1, size_in.x * size_in.y * size_in.z, 1, 1, ecx, 0, gpu->context, queue);
	}
	#endif
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	self->b = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	self->db = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	self->w = Tensor_new(tamanhoSaida, size_in.x * size_in.y * size_in.z, 1, 1, ecx, 0, gpu->context, queue);

	self->dw = Tensor_new(tamanhoSaida, size_in.x * size_in.y * size_in.z, 1, 1, ecx, 0, gpu->context, queue);
	ECX_IF_FAILED(ecx, methods)
	self->dw->fill(self->dw, 0);

	self->db->fill(self->db, 0);

	self->rdp_pesos = rdp_pesos;
	self->rdp_bias = rdp_bias;
	if (rdp_pesos.type != -1) {
		if (rdp_pesos.type < 0) {
			rdp_pesos.type = 0;
		}
		if (rdp_pesos.type == 0) {
			rdp_pesos = internal_getDefaultRDP(self->fa.id == FLRELU, size_in.x * size_in.y * size_in.z, Super.s->length);
			self->rdp_pesos.type = -rdp_pesos.type - 100;
			self->rdp_pesos.a = rdp_pesos.a;
			self->rdp_pesos.b = rdp_pesos.b;
		}
		Super.ecx->error = self->w->randomize(self->w, rdp_pesos.type, rdp_pesos.a, rdp_pesos.b);

	}

	if (rdp_bias.type != -1) {
		if (rdp_bias.type < 0) {
			rdp_bias.type = 0;
		}
		if (rdp_bias.type == 0) {
			if (self->fa.id == FLRELU) {
				self->b->fill(self->b, 0);
			} else {
				rdp_bias = internal_getDefaultRDP(1, size_in.x * size_in.y * size_in.z, Super.s->length);
				Super.ecx->error = self->b->randomize(self->b, rdp_bias.type, rdp_bias.a, rdp_bias.b);

			}
			self->rdp_bias.type = -rdp_bias.type - 100;
			self->rdp_bias.a = rdp_bias.a;
			self->rdp_bias.b = rdp_bias.b;
		}

	}

	self->fa.mask = funcaoDeAtivacao;
	if (self->fa.id == FRELU) {
		self->fa.mask = FATIVACAO(FLRELU, 0, 1);
	}
	self->dfa = self->fa.id | FLAGDIF;

	// Propagação
	{
		COD(DEFAULT_COD)
		COD("__kernel void  feed("
			"__global REAL * a, "
			"__global REAL * w, "
			"__global REAL * b, "
			"__global REAL * z, "
			"int k0")
		if (self->fa.id != FSOFTMAX) {
			COD(",__global REAL * s")
		}

		COD("){\n"
			"\tint m = get_global_id(0) + k0;\n"
			"\tREAL sum = b[m];\n"
			"\tint n;\n"
			"\tfor (n = 0; n < %zu; n++) {\n"
			"\t\tsum += a[n] * w[m*%zu + n];\n"
			"\t}\n"
			"\tz[m] = sum;\n", self->w->y, self->w->y)
		switch (self->fa.id) {
			case FLRELU:
				IFCOD(self->fa.greater == 1.0, "\ts[m] = sum > 0? sum:") ELSECOD("\ts[m] = sum > 0? sum *%.12f:", self->fa.greater)
				IFCOD(self->fa.less == 0.0, "0.0;") ELSECOD("sum*%.12f;", self->fa.less);
				break;
			case FSIGMOID: COD("\ts[m] =  1.0 / (1.0 + exp(-sum));\n")
				break;
			case FTANH: COD("\ts[m] = tanh(sum);")
				break;
			case FLIN: COD("\ts[m] = sum;")
				break;
			case FALAN: COD("\ts[m] = sum >1.0?log10(sum) + 0.7615941559557649:"
							"(sum<-1.0?-log10(-sum) - 0.7615941559557649:tanh(sum));")
				break;
			case FSOFTMAX:
				break;
			default:
				Super.ecx->setError(Super.ecx, GAB_INVALID_PARAM, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
				fprintf(stderr, "Função de ativação desconhecida\n");
				goto methods;
		}
		COD("\n}\n")
		if (self->fa.id == FSOFTMAX) {
			COD("__kernel void calc_exp(__global REAL * exponenciais, __global REAL * z,REAL maximo, int k0){\n"
				"\tint k = get_global_id(0) + k0;\n"
				"\texponenciais[k] = exp(z[k] - maximo);\n"
				"}\n")

			COD("__kernel void ativa(__global REAL *exponenciais, __global REAL *s, REAL sum, int k0){\n"
				"\tint k = get_global_id(0) + k0;\n"
				"\tREAL result =  exponenciais[k]/sum;\n"
				"\ts[k] = result<%.12f?%.12f:(result>%.12f?%.12f:result);\n"
				"}\n", self->fa.epsilon, self->fa.epsilon, 1 - self->fa.epsilon, 1 - self->fa.epsilon)
		}


	}

	// retro propagação
	{
		// calcula dz db e corrige b
		COD("__kernel void calc_dzdb("
			"__global REAL * b, "
			"__global REAL * db, "
			"__global REAL * dz, "
			"__global REAL * ds, "
			"__global REAL * z, "
			"__global REAL * s, "
			"REAL learnRate, "
			"int k0"
			") {\n\tint m = get_global_id(0) + k0;\n");
		switch (self->fa.id) {
			case FLRELU:
				IFCOD(self->fa.greater == 1.0, "\tdz[m] = ds[m] * (z[m] > 0? 1.0:") ELSECOD("\tdz[m] = ds[m] * ( z[m] > 0? %.12f:", self->fa.greater)
				IFCOD(self->fa.less == 0.0, "0.0);") ELSECOD("%.12f);", self->fa.less);
				break;
			case FSIGMOID: COD("\tdz[m] = ds[m] * s[m] * ( 1.0 - s[m]);")
				break;
			case FTANH: COD("\tdz[m] = ds[m] * (1.0 - s[m] * s[m]);")
				break;
			case FLIN: COD("\tdz[m] = ds[m];")
				break;
			case FALAN: COD("\tREAL tmp = z[m] >1.0? 0.419978/ (z[m]):(z[m]<-1.0?-0.419978/ (z[m]): 1 - s[m] * s[m]);\n"
							"\tdz[m] = tmp * ds[m];")
				break;
			case FSOFTMAX: COD("\tdz[m] = ds[m];")
				break;

			default:
				Super.ecx->setError(Super.ecx, GAB_INVALID_PARAM, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
				fprintf(stderr, "Função de ativação desconhecida\n");
				goto methods;
		}
		IFCOD(Super.params.momento != 0, "\ndb[m]  = db[m]*%.8f + dz[m];\n", Super.params.momento) ELSECOD("\n""\tdb[m]  =  dz[m];\n")
		COD("\tCORRIGIR_PESOS(b[m],db[m],learnRate,%f);\n", Super.params.decaimento);
		COD("}\n");
		// calcula dz e db com batch
		COD("__kernel void calc_dzdb_batch("
			"__global REAL * db, "
			"__global REAL * dz, "
			"__global REAL * ds, "
			"__global REAL * z, "
			"__global REAL * s, "
			"long batch_size, "
			"int k0"
			") {\n\tint m = get_global_id(0) + k0;\n");
		switch (self->fa.id) {
			case FLRELU:
				IFCOD(self->fa.greater == 1.0, "\tdz[m] = ds[m]*(z[m] > 0? 1.0:") ELSECOD("\tdz[m] = ds[m]*(z[m] > 0? %.12f:", self->fa.greater)
				IFCOD(self->fa.less == 0.0, "0.0);") ELSECOD("%.12f);", self->fa.less);
				break;
			case FSIGMOID: COD("\tdz[m] = ds[m] * s[m] * ( 1.0 - s[m]);")
				break;
			case FTANH: COD("\tdz[m] = ds[m] * (1.0 - s[m]);")
				break;
			case FLIN: COD("\tdz[m] = ds[m];")
				break;
			case FALAN: COD("\tREAL tmp = z[m] >1.0? 0.419978/ (z[m]):(z[m]<-1.0?-0.419978/ (z[m]): 1 - s[m] * s[m]);\n"
							"\tdz[m] = tmp * ds[m];")
				break;
			case FSOFTMAX: COD("\tdz[m] = ds[m];")
				break;

			default:
				Super.ecx->setError(Super.ecx, GAB_INVALID_PARAM, "Função de ativação desconhecida\n");
				goto methods;
		}
		COD("\n\tdb[m]  = db[m]  +  dz[m]/batch_size;\n")
		COD("}\n");

		COD("__kernel void corrige_peso(__global REAL *w,__global REAL * dw,REAL learnRate, int k0){\n"
			"\tint k = get_global_id(0) + k0;\n"
			"\tCORRIGIR_PESOS(w[k],dw[k],learnRate,%f);\n"
			"\tdw[k] = dw[k] * %f;\n"
			"}\n", Super.params.decaimento, Super.params.momento)

		// calcula da
		COD("__kernel void calc_da(__global REAL * da,__global REAL *w,__global REAL* dz, int k0) {\n "
			"\tint m = get_global_id(0) + k0;\n"
			"\tREAL soma = 0;\n"
			"\tfor (int n = 0; n < %zu; ++n) {\n"
			"\t\tsoma += dz[n] * w[kMap(n, m, 0, %zu, %zu)];\n"
			"\t}\n"
			"\tda[m] = soma;\n}\n",
			self->w->x, self->w->x, self->w->y)


		COD("__kernel void calc_dw(__global REAL *dw,__global REAL *dz,"
			"__global REAL *a,__global REAL *w,REAL learnRate, int k0){\n"
			"\tint k = get_global_id(0) + k0;\n"
			"\tint m, n;\n"
			"\tm = k / %zu;\n"
			"\tn = k %% %zu;\n"
			"\tdw[k] = dz[m] * a[n] + dw[k]*%f;\n"
			"\tCORRIGIR_PESOS(w[k],dw[k],learnRate,%f);\n"
			"}\n", self->w->y, self->w->y, Super.params.momento, Super.params.decaimento)

		COD("__kernel void calc_dw_batch("
			"__global REAL *dw, "
			"__global REAL *dz, "
			"__global REAL *a, "
			"long batch_size, int k0){\n"
			"\tint k = get_global_id(0) + k0;\n"
			"\tint m, n;\n"
			"\tm = k / %zu;\n"
			"\tn = k %% %zu;\n"

			"\tdw[k] = dz[m] * a[n]/batch_size + dw[k];\n"
			"}\n", self->w->y, self->w->y)


	}
	char nametmp[250];
	static int denseL = 0;
	snprintf(nametmp, 250, "layers/dense%d.h", denseL);

	denseL++;
	FILE *ftmp = fopen(nametmp, "w");
	if (ftmp) {
		fprintf(ftmp, "%s", Super.kernel);
		fclose(ftmp);
	}

	internal_compile((Camada) self, gpu);
	ECX_IF_FAILED(Super.ecx, end)
	self->feed = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "feed", Super.ecx->perro);
	if (self->fa.id == FSOFTMAX) {
		self->expoente = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
		self->calc_exp = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_exp", Super.ecx->perro);
		self->ativa = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "ativa", Super.ecx->perro)
	}


	self->calc_dzdb = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_dzdb", Super.ecx->perro)
	self->calc_dzdb_batch = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_dzdb_batch", Super.ecx->perro)
	self->corrige_peso = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "corrige_peso", Super.ecx->perro)
	self->calc_da = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_da", Super.ecx->perro)
	self->calc_dw = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_dw", Super.ecx->perro)
	self->calc_dw_batch = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_dw_batch", Super.ecx->perro)


	methods:
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	Super.release = (void (*)(void *)) CamadaFullConnect_release;
	Super.propagation = (Tensor (*)(void *, Tensor)) (self->fa.id == FSOFTMAX ? CamadaFullConnect_propagation_softmax : CamadaFullConnect_propagation);
	Super.retroPropagation = (int (*)(void *, Tensor)) CamadaFullConnect_backpropagation;
	Super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaFullConnect_backpropagationBatch;
	Super.retroPropagationBatchLearn = (int (*)(void *)) CamadaFullConnect_learnBatch;
	Super.json = (char *(*)(void *, int)) CamadaFullConnect_json;
	Super.getGenerate = (char *(*)(void *)) CamadaFullConnect_getGenerate;
	Super.save = (int (*)(void *, FILE *)) CamadaFullConnect_save;
	Super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaFullConnect_fprintf;
	return (Camada) self;
}

