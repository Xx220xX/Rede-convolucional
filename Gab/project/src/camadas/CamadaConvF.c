//
// Created by hslhe on 19/11/2021.
//


#include "camadas/CamadaConvF.h"

static const char *lname = "ConvolucaoF";


static void CamadaConvF_release(CamadaConvF *self_p) {
	if (!self_p) {
		return;
	}
	if (!*self_p) {
		return;
	}
	internal_Camada_release((Camada *) self_p);
	Release((*self_p)->w);
	Release((*self_p)->dw);
	Release((*self_p)->z);
	Release((*self_p)->dz);
	Release((*self_p)->b);
	Release((*self_p)->db);
	Release((*self_p)->convFSum);
	Release((*self_p)->convFCalcGradIn);
	Release((*self_p)->convFCalcGradBAndFix);
	Release((*self_p)->convFCalcGradBBatch);
	Release((*self_p)->convFCalcGradIn);
	Release((*self_p)->convFCalcGradZ);
	Release((*self_p)->convFCalcGradAndFixWeight);
	Release((*self_p)->convFCalcGradBatch);
	Release((*self_p)->kernel_fixW);
	ReleaseKernel((*self_p)->conv);
	ReleaseKernel((*self_p)->calc_dz);
	ReleaseKernel((*self_p)->calc_da);
	ReleaseKernel((*self_p)->calc_dw);
	ReleaseKernel((*self_p)->calc_dw_batch);
	ReleaseKernel((*self_p)->calc_db);
	ReleaseKernel((*self_p)->calc_db_batch);
	ReleaseKernel((*self_p)->corrige_peso);
	gab_free(*self_p);
	*self_p = NULL;
}


static Tensor CamadaConvF_propagation(CamadaConvF self, Tensor a) {
	Super.a = a;
	setKernelArg(self->conv, 0, void *, Super.a->data);
	setKernelArg(self->conv, 1, void *, self->w->data);
	setKernelArg(self->conv, 2, void *, self->b->data);
	setKernelArg(self->conv, 3, void *, self->z->data);
	setKernelArg(self->conv, 4, void *, Super.s->data);
	runr_kernel(Super.ecx->error, self->conv, Super.s->length, *Super.maxcompute, 5)
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.s;
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	fprintf(stderr, "Error %d\n", Super.ecx->error);
	return NULL;
}

static int CamadaConvF_backpropagation(CamadaConvF self, Tensor ds) {
	// calcula dz
	setKernelArgt(self->calc_dz, 0, self->dz->data);
	setKernelArgt(self->calc_dz, 1, ds->data);
	setKernelArgt(self->calc_dz, 2, self->z->data);
	setKernelArgt(self->calc_dz, 3, Super.s->data);
	runr_kernel(Super.ecx->error, self->calc_dz, self->dz->length, *Super.maxcompute, 4);

	if (Super.da) {
		setKernelArgt(self->calc_da, 0, Super.da->data);
		setKernelArgt(self->calc_da, 1, self->dz->data);
		setKernelArgt(self->calc_da, 2, self->w->data);
		runr_kernel(Super.ecx->error, self->calc_da, Super.da->length, *Super.maxcompute, 3);

	}

	if (!Super.params.skipLearn) {
		setKernelArgt(self->calc_dw, 0, self->w->data);
		setKernelArgt(self->calc_dw, 1, self->dw->data);
		setKernelArgt(self->calc_dw, 2, self->dz->data);
		setKernelArgt(self->calc_dw, 3, Super.a->data);
		setKernelArgt(self->calc_dw, 4, Super.params.hitlearn);
		runr_kernel(Super.ecx->error, self->calc_dw, self->dw->length, *Super.maxcompute, 5);

		setKernelArgt(self->calc_db, 0, self->b->data);
		setKernelArgt(self->calc_db, 1, self->db->data);
		setKernelArgt(self->calc_db, 2, self->dz->data);
		setKernelArgt(self->calc_db, 3, Super.params.hitlearn);
		runr_kernel(Super.ecx->error, self->calc_db, self->db->length, *Super.maxcompute, 4);

	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	fprintf(stderr, "Error %d\n", Super.ecx->error);
	return Super.ecx->error;;
}


static int CamadaConvF_backpropagationBatch(CamadaConvF self, Tensor ds, size_t batchSize) {

	setKernelArgt(self->calc_dz, 0, self->dz->data);
	setKernelArgt(self->calc_dz, 1, ds->data);
	setKernelArgt(self->calc_dz, 2, self->z->data);
	setKernelArgt(self->calc_dz, 3, Super.s->data);
	runr_kernel(Super.ecx->error, self->calc_dz, self->dz->length, *Super.maxcompute, 4);
	ECX_IF_FAILED(Super.ecx, handle_error)
	if (Super.da) {
		setKernelArgt(self->calc_da, 0, Super.da->data);
		setKernelArgt(self->calc_da, 1, self->dz->data);
		setKernelArgt(self->calc_da, 2, self->w->data);
		runr_kernel(Super.ecx->error, self->calc_da, Super.da->length, *Super.maxcompute, 3);
		ECX_IF_FAILED(Super.ecx, handle_error)

	}
	if (!Super.params.skipLearn) {
		setKernelArgt(self->calc_dw_batch, 0, self->dw->data);
		setKernelArgt(self->calc_dw_batch, 1, self->dz->data);
		setKernelArgt(self->calc_dw_batch, 2, Super.a->data);
		setKernelArgt(self->calc_dw_batch, 3, batchSize);
		runr_kernel(Super.ecx->error, self->calc_dw_batch, self->dw->length, *Super.maxcompute, 4);
		ECX_IF_FAILED(Super.ecx, handle_error)

		setKernelArgt(self->calc_db_batch, 0, self->db->data);
		setKernelArgt(self->calc_db_batch, 1, self->dz->data);
		setKernelArgt(self->calc_db_batch, 2, batchSize);
		runr_kernel(Super.ecx->error, self->calc_db_batch, self->db->length, *Super.maxcompute, 3);
		ECX_IF_FAILED(Super.ecx, handle_error)

	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	fprintf(stderr, "Error %d\n", Super.ecx->error);
	return Super.ecx->error;;
}

static int CamadaConvF_learnBatch(CamadaConvF self) {
	if (!Super.params.skipLearn) {
		setKernelArgt(self->corrige_peso, 0, self->w->data);
		setKernelArgt(self->corrige_peso, 1, self->dw->data);
		setKernelArgt(self->corrige_peso, 2, Super.params.hitlearn);
		runr_kernel(Super.ecx->error, self->corrige_peso, self->w->length, *Super.maxcompute, 3);
		ECX_IF_FAILED(Super.ecx, handle_error)

		setKernelArgt(self->corrige_peso, 0, self->b->data);
		setKernelArgt(self->corrige_peso, 1, self->db->data);
		setKernelArgt(self->corrige_peso, 2, Super.params.hitlearn);
		runr_kernel(Super.ecx->error, self->corrige_peso, self->b->length, *Super.maxcompute, 3);
		ECX_IF_FAILED(Super.ecx, handle_error)

	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;
	handle_error:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	fprintf(stderr, "Error %d\n", Super.ecx->error);
	return Super.ecx->error;;
}

static char *CamadaConvF_json(CamadaConvF self, int showValues) {
	ECX_RETURN_IF_ERROR(Super.ecx, NULL)
	char *string = NULL;
	int len = 0;
	char *tmp = internal_json((Camada) self, showValues);
	ECX_IF_FAILED(Super.ecx, end)

	apendstr(string, len, "{"
			PAD"%s,\n"
			PAD"\"functionActivation\":%d,\n"
			PAD"\"passo\":[%zu,%zu],\n"
			PAD"\"numero_filtros\":%zu", tmp, self->fa.id, self->passox, self->passoy, self->w->w);
	gab_free(tmp);
	apendTensor("filtros", w, string, len, tmp, showValues);
	apendTensor("grad_filtros", dw, string, len, tmp, showValues);
	apendstr(string, len, "\n}");
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return string;
}

static char *CamadaConvF_getGenerate(CamadaConvF self) {

	char *string = NULL;
	int len = 0;
	GEN_LAYERNAME(string, len);
	GENN_P2D(P2D(self->passox, self->passoy), string, len);
	GENN_P3D(P3D(self->w->x, self->w->y, self->w->w), string, len);
	internal_putFativacao(&string, &len, self->fa.mask);
	apendstr(string, len, ", ");
	GENN_P2D(P2D(self->pad_top, self->pad_bottom), string, len);
	GENN_P2D(P2D(self->pad_left, self->pad_right), string, len);
	GENN_PARAMS(Super.params, string, len);
	GEN_RDP(self->rdp_filtros, string, len);
	GEN_END(string, len);
//	apendstr(string, len, "%s (P2D(%zu, %zu), P3D(%zu, %zu, %zu), %s, Params(%g, %g, %g, %d), RDP(%d, %g, %g))", lname, self->passox, self->passoy, self->w->x, self->w->y, self->w->w, F_ATIVACAO_NAME(self->fa), (double) Super.params.hitlearn, (double) Super.params.momento, (double) Super.params.decaimento, Super.params.skipLearn, self->rdp_filtros.type, (double) self->rdp_filtros.a, (double) self->rdp_filtros.b);
	return string;
}


static int CamadaConvF_save(CamadaConvF self, FILE *f) {
	ECX_RETURN_IF_ERROR(Super.ecx, Super.ecx->error)
	Super.ecx->addstack(Super.ecx, "CamadaConvF_save");
	internal_saveCamada(f, (Camada) self);
	fwrite(&self->passox, 1, sizeof(size_t), f);
	fwrite(&self->passoy, 1, sizeof(size_t), f);
	fwrite(&self->w->x, 1, sizeof(size_t), f);
	fwrite(&self->w->y, 1, sizeof(size_t), f);
	fwrite(&self->w->w, 1, sizeof(size_t), f);
	fwrite(&self->fa.mask, 1, sizeof(FAtivacao_t), f);
	fwrite(&self->pad_top, 1, sizeof(uint32_t), f);
	fwrite(&self->pad_bottom, 1, sizeof(uint32_t), f);
	fwrite(&self->pad_left, 1, sizeof(uint32_t), f);
	fwrite(&self->pad_right, 1, sizeof(uint32_t), f);
	internal_saveTensor(f, self->w);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	return Super.ecx->error;
}

Camada CamadaConvF_load(FILE *f, Gpu gpu, Queue queue, Tensor entrada, Ecx ecx) {
	ECX_RETURN_IF_ERROR(ecx, NULL)
	Parametros parametros;
	P3d size_in;
	uint32_t size_element;
	FAtivacao_t fativacao;
	P2d passo;
	P3d filtro;
	internal_loadCamada(f, &parametros, &size_in, &size_element);
	fread(&passo.x, sizeof(size_t), 1, f);
	fread(&passo.y, sizeof(size_t), 1, f);
	fread(&filtro.x, sizeof(size_t), 1, f);
	fread(&filtro.y, sizeof(size_t), 1, f);
	fread(&filtro.z, sizeof(size_t), 1, f);
	fread(&fativacao, sizeof(FAtivacao_t), 1, f);
	uint32_t top, bottom, left, right;
	fread(&top, 1, sizeof(uint32_t), f);
	fread(&bottom, 1, sizeof(uint32_t), f);
	fread(&left, 1, sizeof(uint32_t), f);
	fread(&right, 1, sizeof(uint32_t), f);
	CamadaConvF self = (CamadaConvF) CamadaConvF_new(gpu, queue, size_in, entrada, ecx, passo, filtro, fativacao, top, bottom, left, right, parametros, RDP(-1));
	internal_loadTensor(f, self->w, size_element);
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(ecx)
	return (Camada) self;
}

int CamadaConvF_fprintf(CamadaConvF self, FILE *destino, char *format, ...) {
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

Camada CamadaConvF_new(INTERNAL_DEFAULT_ARGS, P2d passo, P3d filtro, FAtivacao_t ativacao, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right, Parametros params, RandomParams rdp_filtros) {
	ECX_RETURN_IF_ERROR(ecx,NULL)
	CamadaConvF self = gab_alloc(1, sizeof(CamadaConvF_t));
	self->pad_top = top;
	self->pad_bottom = bottom;
	self->pad_left = left;
	self->pad_right = right;
	P3d size_out = {(size_in.x + self->pad_top + self->pad_bottom - filtro.x) / passo.x + 1, (size_in.y + self->pad_left + self->pad_right - filtro.y) / passo.y + 1, filtro.z};
	internal_Camada_new((Camada) self, gpu, queue, CONVOLUCAOF_ID, lname, params, entrada, size_in, size_out, ecx);

	self->fa.mask = ativacao;
	if (self->fa.id == FRELU) {
		self->fa.mask = FATIVACAO(FLRELU, 0, 1);
	}
	self->derivationFuntion = ativacao | FLAGDIF;

	self->dw = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->b = Tensor_new(1, 1, filtro.z, 1, ecx, TENSOR3D, gpu->context, queue);
	self->db = Tensor_new(1, 1, filtro.z, 1, ecx, TENSOR3D, gpu->context, queue);
	self->w = Tensor_new(filtro.x, filtro.y, size_in.z, filtro.z, ecx, TENSOR4D, gpu->context, queue);
	self->z = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);
	self->dz = Tensor_new(size_out.x, size_out.y, size_out.z, 1, ecx, 0, gpu->context, queue);

	ECX_IF_FAILED(Super.ecx, end)
	self->dw->fill(self->dw, 0);
	self->b->fill(self->b, 0);
	self->db->fill(self->db, 0);


	self->rdp_filtros = rdp_filtros;

	if (rdp_filtros.type != -1) {
		if (rdp_filtros.type < 0) {
			rdp_filtros.type = 0;
		}
		if (rdp_filtros.type == 0) {
			rdp_filtros = internal_getDefaultRDP(self->fa.id == FLRELU, filtro.x * filtro.y * size_in.z, Super.s->length);
			self->rdp_filtros.type = -rdp_filtros.type - 100;
			self->rdp_filtros.a = rdp_filtros.a;
			self->rdp_filtros.b = rdp_filtros.b;
		}
		Super.ecx->error = self->w->randomize(self->w, rdp_filtros.type, rdp_filtros.a, rdp_filtros.b);

	}
	self->passox = passo.x;
	self->passoy = passo.y;
	// kernel macros
	{
		COD(DEFAULT_COD);
	}
	// kernel de soma de convolução
	{
		COD("__kernel void sum ("
			"__global REAL *A, "
			"__global REAL *weight, "
			"__global REAL *bias, "
			"__global REAL *Z, "
			"__global REAL *S, "
			"int k0){\n""	int k = get_global_id(0) + k0  ;\n"
			"	int x, y, w;\n"
			"	kRap(k, x, y, w, %zu, %zu);\n"
			"	REAL sum, f, v;\n"
			"	int lf, le;\n",
			size_out.x, size_out.y)
		IFCOD(passo.x != 1, "	x = x * %zu;\n", passo.x);
		IFCOD(passo.y != 1, "	y = y * %zu;\n", passo.y);

		COD("	int m0 = 0, mf = %zu;\n"
			"	if(x<%zu){ m0 = %zu;}\n"
			"	if(x+mf >= %zu){mf = %llu - x;}\n",
			filtro.x,
			self->pad_top, self->pad_top,
			self->pad_top + size_in.x, size_in.x + self->pad_top
		   )
		COD("	int n0 = 0, nf = %zu;\n"
			"	if(y<%zu){ n0 = %zu;}\n"
			"	if(y+nf >= %zu){nf = %llu - y ;}\n",
			filtro.y,
			self->pad_left, self->pad_left,
			self->pad_left + size_in.y, size_in.y + self->pad_left
		   )

		COD("	sum = bias[w];\n"
			"	for (int z = 0; z < %zu; z++) {\n"
			"		for (int m = m0; m < mf; m++) {\n"
			"			for (int n = n0; n < nf; n++) {\n"
			"				lf = kMap4D(m, n, z, w, %zu, %zu, %zu);\n"
			"				le = kMap(x  + m - %zu, y + n - %zu, z, %zu, %zu);\n"
			"				f = weight[lf];\n"
			"				v = A[le];\n"
			"				sum += f * v;\n"
			//			"				printf(\"(%%d,%%d,%%d,%%d) = %%.10f , %%.10f,%%.10f\\n\",m,n,z,w,f,v,f*v);"
			"			}\n"
			"		}\n"
			"	}\n"
			"	Z[k] = sum;\n",
			size_in.z, filtro.x, filtro.y, size_in.z,
			self->pad_top, self->pad_left, size_in.x, size_in.y)
		switch (self->fa.id) {
			case FLRELU:
				IFCOD(self->fa.greater == 1.0, "\tS[k] = sum > 0? sum:") ELSECOD("\tS[k] = sum > 0? sum *%.10f:", self->fa.greater)
				IFCOD(self->fa.less == 0.0, "0.0;\n}\n") ELSECOD("sum*%.10f;\n}\n", self->fa.less);
				break;
			case FSIGMOID: COD("\tS[k] =  1.0 / (1.0 + exp(-sum));\n}\n")
				break;
			case FTANH: COD("\tS[k] = tanh(sum);\n}\n")
				break;
			case FLIN: COD("\tS[k] = sum;\n}\n")
				break;
			case FALAN: COD("\tS[k] = sum >1.0?log10(sum) + 0.7615941559557649:"
							"(sum<-1.0?-log10(-sum) - 0.7615941559557649:tanh(sum));\n}\n")
				break;
		}

	}

	// kernel de retro propagação
	{
		COD("typedef struct {\n"
			"\tint x, y, z;\n"
			"} Ponto3d;\n"
			"\n"
			"typedef struct {\n"
			"\tPonto3d min, max;\n"
			"} Range;\n");
		COD("__kernel void calc_dz("
			"__global REAL * dz, "
			"__global REAL * ds, "
			"__global REAL * z, "
			"__global REAL * s, "
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
			default:
				Super.ecx->setError(Super.ecx, GAB_INVALID_PARAM, "%s:%d %s", __FILE__, __LINE__, __FUNCTION__);
				fprintf(stderr, "Função de ativação desconhecida\n");
				goto methods;
		}
		COD("\n}\n")
		COD("__kernel void calc_da("
			"__global REAL *da, "
			"__global REAL *dz, "
			"__global REAL *weight, "
			"int k0"
			"){\n"
			"  int k = get_global_id(0) + k0;\n"
			"  int x, y, z;\n"
			"  kRap(k, x, y, z, %zu, %zu)\n", size_in.x, size_in.y)
		IFCOD(self->pad_top, "  x = x + %zu;", self->pad_top);
		IFCOD(self->pad_left, "  y = y + %zu;", self->pad_left);
		COD("  Range range_W;\n"
			"  range_W.min.x = 0;\n"
			"  if (x + %zu > %zu) {\n"
			"    range_W.min.x = x + %zu - %zu;\n"
			"  }\n", filtro.x, size_in.x + self->pad_top + self->pad_bottom, filtro.x, size_in.x + self->pad_top + self->pad_bottom)

		COD("  range_W.max.x = %zu - 1;\n"
			"  if (x - %zu + 1 < 0) {\n"
			"    range_W.max.x = x;\n"
			"  }\n", filtro.x, filtro.x)

		COD("  range_W.min.y = 0;\n"
			"  if (y + %zu > %zu) {\n"
			"    range_W.min.y = y + %zu - %zu;\n"
			"  }\n", filtro.y, size_in.y + self->pad_left + self->pad_right, filtro.y, size_in.y + self->pad_left + self->pad_right)

		COD("  range_W.max.y = %zu - 1;\n"
			"  if (y - %zu + 1 < 0) {\n"
			"    range_W.max.y = y;\n"
			"  }\n", filtro.y, filtro.y)

		COD("  REAL somaErro = 0, pesoAplicado = 0;\n"
			"  int i, j;\n"
			"  int lf, ls;\n"
			"  for (int m = range_W.min.x; m <= range_W.max.x; m++) {\n"
			"    i = (x - m) / %zu;\n", passo.x)
		COD("    if (i * %zu + m != x) {\n"
			"      continue;\n"
			"    }\n", passo.x)
		COD("    for (int n = range_W.min.y; n <= range_W.max.y; n++) {\n"
			"      j = (y - n) / %zu;\n"
			"      if (j * %zu + n != y) {\n"
			"        continue;\n"
			"      }\n", passo.y, passo.y)
		COD("      for (int w = 0; w < %zu; w++) {\n", size_out.z)
		COD("        lf = kMap4D(m, n, z, w, %zu, %zu, %zu);\n", filtro.x, filtro.y, size_in.z)
		COD("        ls = kMap(i, j, w, %zu, %zu);\n", size_out.x, size_out.y)
		COD("        pesoAplicado = weight[lf];\n"
			"        somaErro += pesoAplicado * dz[ls];\n"
			"      }\n"
			"    }\n"
			"  }\n"
			"  da[k] = somaErro;"
			"\n}\n")

		COD("__kernel void calc_dw("
			"__global REAL * w, "
			"__global REAL * dw, "
			"__global REAL * dz, "
			"__global REAL * a, "
			"REAL learnRate, "
			"int k0){\n"
			"    int k = get_global_id(0) + k0;\n"
			"    int m, n, z, l;\n"
			"    kRep4D(k, m, n, z, l, %zu, %zu, %zu)\n", filtro.x, filtro.y, size_in.z)

		COD("    REAL soma = 0;\n"
			"    int le, ls;\n"
			"    int x,y;\n"
			"    for (int i = 0; i < %zu; ++i) {\n", size_out.x)
		COD("        x = i * %zu + m - %zu;\n"
			"        if(x<0)continue;\n", passo.x, self->pad_top)
		COD("        if(x>=%zu)continue;\n", size_in.x)
		COD("        for (int j = 0; j < %zu; ++j) {\n", size_out.y)
		COD("            y = j * %zu + n - %zu;\n", passo.y, self->pad_left)
		COD("            if(y<0)continue;\n")
		COD("            if(y>=%zu)continue;\n", size_in.y)
		COD("            le = kMap(x, y, z, %zu, %zu);\n", size_in.x, size_in.y)

		COD("            ls = kMap(i, j, l, %zu, %zu);\n", size_out.x, size_out.y)
		COD("            soma += a[le] * dz[ls];\n"
			"        }\n"
			"    }\n"
			"    dw[k] = soma + dw[k] * %.10f;\n", Super.params.momento);
		COD("    CORRIGIR_PESOS(w[k],dw[k],learnRate,%.10f);\n}\n", Super.params.decaimento)

		COD("__kernel void calc_dw_batch("
			"__global REAL * dw, "
			"__global REAL * dz, "
			"__global REAL * a, "
			"long batch_size, "
			"int k0){\n"
			"    int k = get_global_id(0) + k0;\n"
			"    int m, n, z, l;\n"
			"    kRep4D(k, m, n, z, l, %zu, %zu, %zu)\n", filtro.x, filtro.y, size_in.z)

		COD("    REAL soma = 0;\n"
			"    int le, ls;\n"
			"    int x,y;\n"
			"    for (int i = 0; i < %zu; ++i) {\n", size_out.x)
		COD("        x = i * %zu + m - %zu;\n"
			"        if(x<0)continue;\n", passo.x, self->pad_top)
		COD("        if(x>=%zu)continue;\n", size_in.x)
		COD("        for (int j = 0; j < %zu; ++j) {\n", size_out.y)
		COD("            y = j * %zu + n - %zu;\n", passo.y, self->pad_left)
		COD("            if(y<0)continue;\n")
		COD("            if(y>=%zu)continue;\n", size_in.y)
		COD("            le = kMap(x, y, z, %zu, %zu);\n", size_in.x, size_in.y)

		COD("            ls = kMap(i, j, l, %zu, %zu);\n", size_out.x, size_out.y)
		COD("            soma += a[le] * dz[ls];\n"
			"        }\n"
			"    }\n"
			"    dw[k] = soma/batch_size + dw[k] ;\n"
			"}\n");


		COD("__kernel void calc_db("
			"__global REAL *b, "
			"__global REAL *dB, "
			"__global REAL *dZ, "
			"REAL learnRate, "
			"int k0"
			"){\n"
			"	int w = get_global_id(0) + k0;"
			"	REAL sum = 0;\n"
			"	for (int x = 0; x < %zu; ++x) {\n"
			"		for (int y = 0; y < %zu; ++y) {\n", self->dz->x, self->dz->y)
		COD("			sum += dZ[kMap(x, y, w, %zu, %zu)];\n", self->dz->x, self->dz->y)
		COD("		}\n	}\n"
			"	dB[w] = sum + dB[w] * %.10f;", Super.params.momento)
		COD("	CORRIGIR_PESOS(b[w],dB[w],learnRate,%f);\n}\n", Super.params.decaimento)

		COD("__kernel void calc_db_batch("
			"__global REAL *dB, "
			"__global REAL *dZ, "
			"long batchSize, "
			"int k0"
			"){\n"
			"	int w = get_global_id(0) + k0;"
			"	REAL sum = 0;\n"
			"	for (int x = 0; x < %zu; ++x) {\n"
			"		for (int y = 0; y < %zu; ++y) {\n", self->dz->x, self->dz->y)
		COD("			sum += dZ[kMap(x, y, w, %zu, %zu)];\n", self->dz->x, self->dz->y)
		COD("		}\n	}\n"
			//			"printf(\"kernel: batchSize %%f\\n\",sum);"
			"	dB[w] = sum/batchSize + dB[w] ;"
			"\n}\n")

		COD("__kernel void corrige_peso(__global REAL *w, __global REAL *dw, REAL learnRate, int k0){\n"
			"	int k = get_global_id(0) + k0;\n"
			"	CORRIGIR_PESOS(w[k],dw[k],learnRate,%f);\n"
			"	dw[k] = dw[k]*%.10f;\n"
			"\n}\n", Super.params.decaimento, Super.params.momento)

	}
	char nametmp[250];
	static int denseL = 0;

	snprintf(nametmp, 250, "layers/conv%d.h", denseL);
	denseL++;
	FILE *ftmp = fopen(nametmp, "w");
	if (ftmp) {
		fprintf(ftmp, "%s", Super.kernel);
		fclose(ftmp);
	}
	internal_compile((Camada) self, gpu);
	ECX_IF_FAILED(Super.ecx, end)
	self->conv = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "sum", Super.ecx->perro);
	self->calc_dz = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_dz", Super.ecx->perro);
	self->calc_da = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_da", Super.ecx->perro);
	self->calc_dw = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_dw", Super.ecx->perro);
	self->calc_dw_batch = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_dw_batch", Super.ecx->perro);
	self->calc_db = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_db", Super.ecx->perro);
	self->calc_db_batch = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "calc_db_batch", Super.ecx->perro);
	self->corrige_peso = ECXCHECKAFTER(Super.ecx, methods, clCreateKernel, Super.program, "corrige_peso", Super.ecx->perro);


	methods:
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(Super.ecx)
	Super.release = (void (*)(void *)) CamadaConvF_release;
	Super.propagation = (Tensor (*)(void *, Tensor)) CamadaConvF_propagation;
	Super.retroPropagation = (int (*)(void *, Tensor)) CamadaConvF_backpropagation;
	Super.retroPropagationBatch = (int (*)(void *, Tensor, size_t)) CamadaConvF_backpropagationBatch;
	Super.retroPropagationBatchLearn = (int (*)(void *)) CamadaConvF_learnBatch;
	Super.json = (char *(*)(void *, int)) CamadaConvF_json;
	Super.getGenerate = (char *(*)(void *)) CamadaConvF_getGenerate;
	Super.save = (int (*)(void *, FILE *)) CamadaConvF_save;
	Super.fprint = (int (*)(void *, FILE *, char *, ...)) CamadaConvF_fprintf;
	return (Camada) self;
}


