//
// Created by Henrique on 02/12/2021.
//

#include "setup/Setup.h"

#include <windows.h>
#include <unistd.h>
#include <math.h>
#include "conio2/conio2.h"
#include "error_list.h"


#define alpha 0.999
#define beta  (1-alpha)

#if (MANAGE_DEBUG_LOAD_LUA == 1)
#define SET_DEBUG(format,...)printf(format,##__VA_ARGS__)
#else
#define SET_DEBUG
#endif
#define SETUP_GETLUA_INT(cvar, name)lua_getglobal(L,name); if(!lua_isnone(L,-1))  (cvar) = lua_tonumber(L,-1); \
else fprintf(stderr,"warning: %s não instanciado em lua\n",name);\
SET_DEBUG("%s %d\n",#cvar,cvar);


#define SETUP_GETLUA_INTE(cvar, name, e)lua_getglobal(L,name); if(!lua_isnoneornil(L,-1))  (cvar) = lua_tonumber(L,-1); \
else {fprintf(stderr,"warning: %s não instanciado em lua\n",name);\
SET_DEBUG("%s %d\n",#cvar,cvar);e}

#define SETUP_GETLUA_BOLL(cvar, name)lua_getglobal(L,name);   \
if(!lua_isnoneornil(L,-1)&&lua_isboolean(L,-1)) (cvar) = lua_toboolean(L,-1); \
else fprintf(stderr,"warning: %s não instanciado em lua\n",name);\
SET_DEBUG("%s %d\n",#cvar,cvar);

#define SETUP_GETLUA_STR(cvar, name)lua_getglobal(L,name);if(luaL_checkstring(L,-1)){                                 \
Free(cvar);                                                                                                                           \
const char *__tmp__ = lua_tostring(L,-1);                                                                                        \
size_t len  =  strlen(__tmp__);                                                                                            \
 (cvar) = gab_alloc(len+1,1);                                                                                              \
 memcpy(cvar,__tmp__,len);\
} else fprintf(stderr,"warning: %s não instanciado em lua\n",name);                                                        \
SET_DEBUG("%s %s\n",#cvar,cvar);

char *asprintf(size_t *tlen, const char *format, ...) {
	size_t len = 0;
	char *tmp = NULL;
	va_list v;
	va_start(v, format);
	len = vsnprintf(NULL, 0, format, v) + 1;
	tmp = gab_alloc(len, 1);
	vsnprintf(tmp, len, format, v) + 1;
	va_end(v);
	if (tlen) {
		*tlen = len;
	}
	return tmp;
}

double seconds() {
	FILETIME ft;
	LARGE_INTEGER li;
	GetSystemTimePreciseAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	u_int64 ret = li.QuadPart;
	return (double) ret * 1e-7;
}

void Setup_checkStop(Setup self, const char *stopPattern) {
	int c = 0;
	if (kbhit()) {
		c = getche();
		for (int i = 0; stopPattern[i]; ++i) {
			if (c == stopPattern[i]) {
				self->can_run = 0;
				return;
			}
		}
	}
}

int Setup_release(Setup *selfp) {
	if (!selfp) {
		return 1;
	}
	if (!*(selfp)) {
		return 2;
	}

	for (int i = 0; i < (*selfp)->n_imagens; ++i) {
		if ((*selfp)->imagens) {
			Release((*selfp)->imagens[i]);
		}
		if ((*selfp)->targets) {
			Release((*selfp)->targets[i]);
		}
	}
	Free((*selfp)->imagens);
	Free((*selfp)->targets);
	Release((*selfp)->labels);
	int erro = (*selfp)->cnn->ecx->error;
	Release((*selfp)->cnn);

	Free((*selfp)->nome);
	Free((*selfp)->home);
	Free((*selfp)->nome_classes);
	Free((*selfp)->file_label);
	Free((*selfp)->file_image);
	Free((*selfp)->treino_out);
	Free((*selfp)->teste_out);
	Free((*selfp)->rede_out);
	if ((*selfp)->te.tce) {
		for (int i = 0; i < (*selfp)->te.nclasses; ++i) {
			Free((*selfp)->te.tce[i].answers);
		}
		Free((*selfp)->te.tce);
	}

	Free(*selfp);
	return erro;
}

void Setup_loadImagens(Setup self) {
	if (!self->file_image) {
		return;
	}
	P3d im_dim = self->cnn->size_in; // dimensao da entrada
	FILE *f; // arquivo para leitura das imagens
	self->iLoad.imTotal = self->n_imagens;// atualiza o iload para leitura em outra thread
	Tensor imgpuReal; // tensor com imagem em valores IR¹
	f = fopen(self->file_image, "rb"); // abre o aquivo
	Tensor imgpu; // imagem 8 bits na gpu
	Tensor imram = Tensor_new(unP3D(im_dim), self->n_imagens, self->cnn->ecx, TENSOR_RAM | TENSOR_CHAR | TENSOR4D); // imagem no hos
	for (int i = 0; i < self->header_image; ++i) {
		fgetc(f);
	}// pula o cabeçalho
	fread(imram->data, 1, imram->bytes, f);// le a imagem
	fclose(f);// fecha o arquivo
	imgpu = Tensor_new(unP3D(im_dim), self->n_imagens, self->cnn->ecx, TENSOR_CHAR | TENSOR4D | TENSOR_CPY, self->cnn->gpu->context, self->cnn->queue, imram->data); // instancia a imagem na gpu
	imgpuReal = Tensor_new(unP3D(im_dim), self->n_imagens, self->cnn->ecx, TENSOR4D, self->cnn->gpu->context, self->cnn->queue, imram->data);// instancia a imagem real
	self->cnn->normalizeIMAGE(self->cnn, imgpuReal, imgpu);// calcula imagem[i]/255
	Release(imram);// libera a imagem da ram
	Release(imgpu);// libera a imagem da gpu

	self->imagens = gab_alloc(self->n_imagens, sizeof(Tensor));// intancia vetor de imagens
	if (self->use_gpu) {
		for (int i = 0; i < self->n_imagens; ++i) { // itera nas imagens
			self->iLoad.imAtual = i + 1; // atualiza status
			self->imagens[i] = Tensor_new(unP3D(im_dim), 1, self->cnn->ecx, 0, self->cnn->gpu->context, self->cnn->queue);//instancia novo tensor
			self->imagens[i]->copyM(self->imagens[i], imgpuReal, 0, i * self->imagens[i]->bytes, self->imagens[i]->bytes);// fz a copia da imagem para o tensor
		}
	} else {
		for (int i = 0; i < self->n_imagens; ++i) { // itera nas imagens
			self->iLoad.imAtual = i + 1; // atualiza status
			self->imagens[i] = Tensor_new(unP3D(im_dim), 1, self->cnn->ecx, TENSOR_RAM);//instancia novo tensor
			self->imagens[i]->copyM(self->imagens[i], imgpuReal, 0, i * self->imagens[i]->bytes, self->imagens[i]->bytes);// fz a copia da imagem para o tensor
		}
	}
	Release(imgpuReal);// libera o tensor imreal
	self->runing = 0; // terminou a leitura de imagens
}

void Setup_loadLabel(Setup self) {
	if (!self->file_label) {
		return;
	}
	self->iLoad.imTotal = self->n_imagens;
	// ### variaveis
	FILE *f = fopen(self->file_label, "rb");
	Tensor lbram = Tensor_new(1, self->n_imagens, 1, 1, self->cnn->ecx, TENSOR_RAM | TENSOR_CHAR, self->cnn->gpu->context, self->cnn->queue);
	Tensor lbgpuREAL = Tensor_new(1, self->n_classes, 1, self->n_imagens, self->cnn->ecx, TENSOR4D, self->cnn->gpu->context, self->cnn->queue);
	Tensor lbgpu;
	// ### ler cabeçalho do arquivo
	for (int i = 0; i < self->header_label; ++i) {
		fgetc(f);
	}
	fread(lbram->data, 1, lbram->bytes, f);
	fclose(f);
	// ### copia para gpu e converter para vetor
	lbgpu = Tensor_new(1, self->n_imagens, 1, 1, self->cnn->ecx, TENSOR_CPY | TENSOR_CHAR, self->cnn->gpu->context, self->cnn->queue, lbram->data);
	self->cnn->extractVectorLabelClass(self->cnn, lbgpuREAL, lbgpu);
	self->targets = gab_alloc(sizeof(Tensor), self->n_imagens);
	// ### separar imagens
	if (self->use_gpu) {
		for (int i = 0; i < self->n_imagens; ++i) {
			self->iLoad.imAtual = i + 1;
			self->targets[i] = Tensor_new(1, self->n_classes, 1, 1, self->cnn->ecx, 0, self->cnn->gpu->context, self->cnn->queue);
			self->targets[i]->copyM(self->targets[i], lbgpuREAL, 0, i * self->targets[i]->bytes, self->targets[i]->bytes);
		}
	} else {
		for (int i = 0; i < self->n_imagens; ++i) {
			self->iLoad.imAtual = i + 1;
			self->targets[i] = Tensor_new(1, self->n_classes, 1, 1, self->cnn->ecx, TENSOR_RAM);
			self->targets[i]->copyM(self->targets[i], lbgpuREAL, 0, i * self->targets[i]->bytes, self->targets[i]->bytes);
		}
	}
	Release(lbgpuREAL);
	Release(lbgpu);
	self->labels = lbram;
	self->runing = 0;
}
float cost_crossEntropy(Tensor S,Tensor T){
	REAL * vS = S->getvalues(S,NULL);
	REAL * vT = T->getvalues(T,NULL);
	REAL sum = 0;
	for (int i = 0; i < S->length; ++i) {
		sum -= vT[i]* log10(vS[i]);
	}
	return sum;
}
REAL cost_mse(Tensor S,Tensor T){
	REAL * vS = S->getvalues(S,NULL);
	REAL * vT = T->getvalues(T,NULL);
	REAL sum = 0;
	REAL tmp;
	for (int i = 0; i < S->length; ++i) {
		tmp = (vS[i] - vT[i]);
		sum += tmp*tmp;
	}
	return sum/S->length;
}

void Setup_treinar(Setup self) {
	ECXPUSH(self->cnn->ecx);
	// ###  thread de alta prioridade
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	// ### variaveis usadas no treino
	Tensor entrada, target;
	ubyte label, cnn_label;

	self->itrain.epTotal = self->n_epocas;
	self->itrain.imTotal = self->n_imagens_treinar;
//	self->itrain.imps = 1e-14;
	Itrain localItrain = self->itrain;
	int indice = 0;
	int classe = 0;
	int images = 0;
	int acertos = 0;
	int acertosep;
	REAL (*cost)(Tensor, Tensor);
	cost = cost_mse;
	if(self->cnn->cm[self->cnn->l-1]->layer_id == SOFTMAX_ID){
		cost = cost_crossEntropy;
	}
	localItrain.winRateMedio = 10;
	localItrain.mse = 1;

	for (; self->can_run && self->epoca_atual < self->n_epocas && !self->cnn->ecx->error; self->epoca_atual++) {
		if (self->imagem_atual_treino >= self->n_imagens_treinar) {
			self->imagem_atual_treino = 0;
		}
		localItrain.epAtual = self->epoca_atual + 1;
		acertosep = 0;
		for (; self->can_run && self->imagem_atual_treino < self->n_imagens_treinar && !self->cnn->ecx->error; self->imagem_atual_treino++) {
			images++;
			indice = self->imagem_atual_treino;
			entrada = self->imagens[indice];
			target = self->targets[indice];
			label = ((char *) self->labels->data)[indice];
			self->cnn->predict(self->cnn, entrada);
			self->cnn->learn(self->cnn, target);
			cnn_label = self->cnn->maxIndex(self->cnn);
			if (self->on_train) {
				self->on_train(self, label);
			}

			// #### informações do treinamento
			acertos += (cnn_label == label);
			acertosep += (cnn_label == label);
			localItrain.winRate = localItrain.winRate * alpha + 100 * beta*(cnn_label == label);
			localItrain.winRateMedio = acertos * 100.0 / (images + 1.0);
			localItrain.winRateMedioep = acertosep * 100.0 / (self->imagem_atual_treino  + 1.0);

			double mse = cost(self->cnn->cm[self->cnn->l-1]->s,target);
			if (isnan(mse)) {
				self->can_run = 0;
				self->force_end = 1;
				self->cnn->ecx->error = GAB_INVALID_PARAM;
				fprintf(stderr, "Erro interno das camadas, encontrado nan\n");
			}
			localItrain.mse = localItrain.mse * alpha + beta * mse;
			localItrain.imAtual = self->imagem_atual_treino + 1;
			self->itrain = localItrain;
			classe++;
		}
	}
	ECXPOP(self->cnn->ecx);
	self->runing = 0;
}

BOOL DirectoryExists(LPCTSTR szPath) {
	DWORD dwAttrib = GetFileAttributes(szPath);

	return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

void Setup_treinarBatch(Setup self) {
	ECXPUSH(self->cnn->ecx);
	// ###  thread de alta prioridade
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	// ### variaveis usadas no treino
	Tensor entrada, target;
	ubyte label, cnn_label;

	self->itrain.epTotal = self->n_epocas;
	self->itrain.imTotal = self->n_imagens_treinar;
	fflush(stderr);
	fflush(stdout);
	Itrain localItrain = self->itrain;
	int indice = 0;
	int classe = 0;
	size_t images = 0;
	size_t iter = 0;
	int acertos = 0;
	REAL (*cost)(Tensor, Tensor);
	cost = cost_mse;
	if(self->cnn->cm[self->cnn->l-1]->layer_id == SOFTMAX_ID){
		cost = cost_crossEntropy;
	}
	size_t bathS;
	char buff[250];

	localItrain.mse = 1;
	localItrain.winRateMedio = 10;
	int acertosep;
	for (; self->can_run && self->epoca_atual < self->n_epocas && !self->cnn->ecx->error; self->epoca_atual++) {
		self->batch = 0;
		bathS = self->n_imagens_treinar >= self->batchSize ? self->batchSize : self->n_imagens_treinar;
		localItrain.epAtual = self->epoca_atual + 1;
		acertosep = 0;
		for (self->imagem_atual_treino = 0; self->can_run && self->imagem_atual_treino < self->n_imagens_treinar && !self->cnn->ecx->error; self->imagem_atual_treino++) {
			images++;
			self->batch++;
			indice = self->imagem_atual_treino;
			entrada = self->imagens[indice];
			target = self->targets[indice];
			label = ((char *) self->labels->data)[indice];
			self->cnn->predict(self->cnn, entrada);
			self->cnn->learnBatch(self->cnn, target, bathS);

			if (self->batch >= self->batchSize) {
				self->cnn->updateHitLearn(self->cnn, iter);
				self->cnn->fixBatch(self->cnn);
				iter++;
				self->batch = 0;
				bathS = self->n_imagens_treinar - self->imagem_atual_treino - 2 >= self->batchSize ? self->batchSize : self->n_imagens_treinar - self->imagem_atual_treino - 1;

			}
			cnn_label = self->cnn->maxIndex(self->cnn);
			if (self->on_train) {
				self->on_train(self, label);
			}

			// #### informações do treinamento
			acertos += (cnn_label == label);
			acertosep += (cnn_label == label);
			localItrain.winRate = localItrain.winRate * alpha + 100 *(cnn_label == label)* beta;
			localItrain.winRateMedio = acertos * 100.0 / (images + 1.0);
			localItrain.winRateMedioep = acertosep * 100.0 / (self->imagem_atual_treino  + 1.0);

			double mse = cost(self->cnn->cm[self->cnn->l-1]->s,target);


			if (isnan(mse)) {
				self->can_run = 0;
				self->force_end = 1;
				self->cnn->ecx->error = GAB_INVALID_PARAM;
				fprintf(stderr, "Erro interno das camadas, encontrado nan\nImagem %zu\nEpoca %d index %d\n", images, self->epoca_atual, indice);
			}
			localItrain.mse = localItrain.mse * alpha + beta * mse;
			localItrain.imAtual = self->imagem_atual_treino + 1;
			self->itrain = localItrain;
			classe++;
		}
		if (self->batch > 0) {
			self->cnn->updateHitLearn(self->cnn, iter);
			self->cnn->fixBatch(self->cnn);
			iter++;
		}
	}
	ECXPOP(self->cnn->ecx);
	self->runing = 0;
}


void TE_new(Setup self) {
	if (self->te.tce != NULL) {
		return;
	}
	self->te.nclasses = self->n_classes;
	self->te.tce = gab_alloc(self->n_classes, sizeof(TesteClasseEstatisticas));
	for (int i = 0; i < self->te.nclasses; ++i) {
		self->te.tce[i].classe = i;
		self->te.tce[i].answers = gab_alloc(self->n_classes, sizeof(int));
	}
}

#define PUSHTE(label, cnnlabel)self->te.tce[label].answers[cnnlabel]++

void Setup_avaliar(Setup self) {
	// ###  thread de alta prioridade
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	// ### variaveis usadas na avaliação
	Tensor entrada, target;
	ubyte label, cnn_label;
	int indice;
	TE_new(self);
	self->iteste.imTotal = self->n_imagens_testar;
	Iteste localIteste = self->iteste;
	localIteste.mse = 0.5;
	int64_t acertos = 0;
	for (; self->can_run && self->imagem_atual_teste < self->n_imagens_testar; self->imagem_atual_teste++) {
		indice = self->n_imagens_treinar + self->imagem_atual_teste;
		entrada = self->imagens[indice];
		target = self->targets[indice];
		label = ((char *) self->labels->data)[indice];

		self->cnn->predict(self->cnn, entrada);
		cnn_label = self->cnn->maxIndex(self->cnn);
		PUSHTE(label, cnn_label);
		acertos += (cnn_label == label);
		localIteste.imAtual = self->imagem_atual_teste + 1;
		localIteste.meanwinRate = 100.0*acertos / (self->imagem_atual_teste + 1.0);
		localIteste.winRate = localIteste.winRate * alpha + beta * ((cnn_label == label) ? 100 : 0);
		localIteste.mse = localIteste.mse * alpha + beta * self->cnn->mseT(self->cnn, target);
		self->iteste = localIteste;
	}
	self->runing = 0;
}

void Setup_saveStatistic(Setup self) {
	char *file_name = asprintf(NULL, "%s.csv", self->nome);
	FILE *file = fopen(file_name, "w");
	int k;
	// colocar todos nomes
	int j;
	int len = strlen(self->nome_classes);
	for (int i = 0; i < self->te.nclasses; ++i) {
		k = 0;
		for (; j < len; j++) {
			if (self->nome_classes[j] == ' ') {
				self->te.tce[i].name[k] = 0;
				j++;
				break;
			} else {
				self->te.tce[i].name[k] = self->nome_classes[j];
				k++;
			}
		}
	}
	fprintf(file, "classe");

	for (int i = 0; i < self->te.nclasses; ++i) {
		fprintf(file, ",%s", self->te.tce[i].name);
	}
	for (int i = 0; i < self->te.nclasses; ++i) {
		fprintf(file, "\n%s", self->te.tce[i].name);
		for (int l = 0; l < self->te.nclasses; ++l) {
			fprintf(file, ",%d", self->te.tce[i].answers[l]);
		}
	}
	fclose(file);
	Free(file_name);
}

void Setup_getLuaParams(Setup self) {
	ECXPUSH(self->cnn->ecx);
	if (!self->cnn) {
		self->cnn->ecx->setError(self->cnn->ecx, GAB_NULL_POINTER_ERROR);
		return;
	}
	if (self->cnn->ecx->error) {
		return;
	}
	lua_State *L = self->cnn->LuaVm;
	if (!L) {
		return;
	}
	SETUP_GETLUA_INT(self->n_epocas, "Numero_epocas");
	SETUP_GETLUA_STR(self->home, "home");
	SETUP_GETLUA_STR(self->nome, "nome");
	SETUP_GETLUA_INT(self->n_imagens, "Numero_Imagens");
	SETUP_GETLUA_INT(self->n_imagens_treinar, "Numero_ImagensTreino");
	SETUP_GETLUA_INT(self->n_imagens_testar, "Numero_ImagensAvaliacao");
	SETUP_GETLUA_INT(self->n_classes, "Numero_Classes");
	SETUP_GETLUA_STR(self->nome_classes, "classes");
	SETUP_GETLUA_INT(self->header_image, "bytes_remanessentes_imagem");
	SETUP_GETLUA_INT(self->header_label, "bytes_remanessentes_classes");
	SETUP_GETLUA_BOLL(self->use_gpu, "gpu_mem");
	SETUP_GETLUA_BOLL(self->useBatch, "use_batch");
	if (self->useBatch) {
		SETUP_GETLUA_INTE(self->batchSize, "batch_size", self->cnn->ecx->addstack(self->cnn->ecx, "get batch_size");self->cnn->ecx->error = GAB_INVALID_PARAM;);
		if (!self->cnn->ecx->error && self->batchSize <= 0) {
			self->cnn->ecx->addstack(self->cnn->ecx, "get batch_size");
			self->cnn->ecx->error = GAB_INVALID_PARAM;
			fprintf(stderr, "O valor não pode ser 0\n");
		}
	}

	SETUP_GETLUA_STR(self->treino_out, "estatisticasDeTreino");
	SETUP_GETLUA_STR(self->teste_out, "estatiscasDeAvaliacao");
	SETUP_GETLUA_STR(self->file_image, "arquivoContendoImagens");
	SETUP_GETLUA_STR(self->file_label, "arquivoContendoRespostas");

	ECXPOP(self->cnn->ecx);
}

#define PFIELD(FIELD, TYPE, f, setup)fprintf(f,#FIELD " = "TYPE"\n",setup->FIELD)

void Setup_loadLua(Setup self, const char *lua_file) {
	ECXPUSH(self->cnn->ecx);
	if (!lua_file) {
		self->cnn->ecx->setError(self->cnn->ecx, GAB_NULL_POINTER_ERROR);
		return;
	}
	CnnLuaLoadFile(self->cnn, lua_file);
	Setup_getLuaParams(self);
	if (self->home) {
		SetCurrentDirectoryA(self->home);
	}
	ECXPOP(self->cnn->ecx);
	// log
	FILE *f = fopen("setup.log", "w");
	PFIELD(n_epocas, "%u", f, self);
	PFIELD(home, "'%s'", f, self);
	PFIELD(nome, "'%s'", f, self);
	PFIELD(n_imagens, "%u", f, self);
	PFIELD(n_imagens_treinar, "%u", f, self);
	PFIELD(n_imagens_testar, "%u", f, self);
	PFIELD(n_classes, "%u", f, self);
	PFIELD(nome_classes, "'%s'", f, self);
	PFIELD(header_image, "%u", f, self);
	PFIELD(use_gpu, "%d", f, self);
	PFIELD(header_label, "%u", f, self);
	PFIELD(treino_out, "'%s'", f, self);
	PFIELD(teste_out, "'%s'", f, self);
	PFIELD(file_image, "'%s'", f, self);
	PFIELD(file_label, "'%s'", f, self);


	fclose(f);
}

int Setup_ok(Setup self) {
	return !(self->force_end || self->cnn->ecx->error);
}

Setup Setup_new() {
	Setup setup = gab_alloc(sizeof(Setup_t), 1);
	setup->can_run = 1;
	setup->cnn = Cnn_new();
	setup->checkStop = Setup_checkStop;
	setup->loadImagens = Setup_loadImagens;
	setup->loadLabels = Setup_loadLabel;
	setup->treinar = Setup_treinar;
	setup->treinarBatch = Setup_treinarBatch;
	setup->avaliar = Setup_avaliar;
	setup->release = Setup_release;
	setup->loadLua = Setup_loadLua;
	setup->saveStatistic = Setup_saveStatistic;
	setup->ok = Setup_ok;
	return setup;
}
