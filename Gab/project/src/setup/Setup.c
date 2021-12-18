//
// Created by Henrique on 02/12/2021.
//

#include "setup/Setup.h"

#include <windows.h>
#include <unistd.h>
#include "conio2/conio2.h"
#include "error_list.h"


#define alpha 0.99
#define beta  (1-alpha)

#if (MANAGE_DEBUG_LOAD_LUA == 1)
#define SET_DEBUG(format,...)printf(format,##__VA_ARGS__)
#else
#define SET_DEBUG
#endif
#define SETUP_GETLUA_INT(cvar, name)lua_getglobal(L,name);   \
if(!lua_isnoneornil(L,-1)&&luaL_checkinteger(L,-1))    (cvar) = lua_tonumber(L,-1); \
else fprintf(stderr,"warning: %s não instanciado em lua\n",name);\
SET_DEBUG("%s %d\n",#cvar,cvar);

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
	if (!selfp) { return 1; }
	if (!*(selfp)) { return 2; }

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
	int erro = (*selfp)->cnn->erro->error;
	Release((*selfp)->cnn);

	Free((*selfp)->nome);
	Free((*selfp)->home);
	Free((*selfp)->nome_classes);
	Free((*selfp)->file_label);
	Free((*selfp)->file_image);
	Free((*selfp)->treino_out);
	Free((*selfp)->teste_out);
	Free((*selfp)->rede_out);
	Free(*selfp);
	return erro;
}

void Setup_loadImagens(Setup self) {
	if (!self->file_image) { return; }
	P3d im_dim = self->cnn->size_in; // dimensao da entrada
	FILE *f; // arquivo para leitura das imagens
	self->iLoad.imTotal = self->n_imagens;// atualiza o iload para leitura em outra thread
	Tensor imgpuReal; // tensor com imagem em valores IR¹
	f = fopen(self->file_image, "rb"); // abre o aquivo
	Tensor imgpu; // imagem 8 bits na gpu
	Tensor imram = Tensor_new(unP3D(im_dim), self->n_imagens, self->cnn->erro, TENSOR_RAM | TENSOR_CHAR | TENSOR4D); // imagem no hos
	for (int i = 0; i < self->header_image; ++i) { fgetc(f); }// pula o cabeçalho
	fread(imram->data, 1, imram->bytes, f);// le a imagem
	fclose(f);// fecha o arquivo
	imgpu = Tensor_new(unP3D(im_dim), self->n_imagens, self->cnn->erro, TENSOR_CHAR | TENSOR4D | TENSOR_CPY, self->cnn->gpu->context, self->cnn->queue, imram->data); // instancia a imagem na gpu
	imgpuReal = Tensor_new(unP3D(im_dim), self->n_imagens, self->cnn->erro, TENSOR4D, self->cnn->gpu->context, self->cnn->queue, imram->data);// instancia a imagem real
	self->cnn->normalizeIMAGE(self->cnn, imgpuReal, imgpu);// calcula imagem[i]/255
	Release(imram);// libera a imagem da ram
	Release(imgpu);// libera a imagem da gpu

	self->imagens = gab_alloc(self->n_imagens, sizeof(Tensor));// intancia vetor de imagens
	for (int i = 0; i < self->n_imagens; ++i) { // itera nas imagens
		self->iLoad.imAtual = i + 1; // atualiza status
		self->imagens[i] = Tensor_new(unP3D(im_dim), 1, self->cnn->erro, 0, self->cnn->gpu->context, self->cnn->queue);//instancia novo tensor
		self->imagens[i]->copyM(self->imagens[i], imgpuReal, 0, i * self->imagens[i]->bytes, self->imagens[i]->bytes);// fz a copia da imagem para o tensor
	}
	Release(imgpuReal);// libera o tensor imreal
	self->runing = 0; // terminou a leitura de imagens
}

void Setup_loadLabel(Setup self) {
	if (!self->file_label) { return; }
	self->iLoad.imTotal = self->n_imagens;
	// ### variaveis
	FILE *f = fopen(self->file_label, "rb");
	Tensor lbram = Tensor_new(1, self->n_imagens, 1, 1, self->cnn->erro, TENSOR_RAM | TENSOR_CHAR, self->cnn->gpu->context, self->cnn->queue);
	Tensor lbgpuREAL = Tensor_new(1, self->n_classes, 1, self->n_imagens, self->cnn->erro, TENSOR4D, self->cnn->gpu->context, self->cnn->queue);
	Tensor lbgpu;
	// ### ler cabeçalho do arquivo
	for (int i = 0; i < self->header_label; ++i) { fgetc(f); }
	fread(lbram->data, 1, lbram->bytes, f);
	fclose(f);
	// ### copia para gpu e converter para vetor
	lbgpu = Tensor_new(1, self->n_imagens, 1, 1, self->cnn->erro, TENSOR_CPY | TENSOR_CHAR, self->cnn->gpu->context, self->cnn->queue, lbram->data);
	self->cnn->extractVectorLabelClass(self->cnn, lbgpuREAL, lbgpu);
	self->targets = gab_alloc(sizeof(Tensor), self->n_imagens);
	// ### separar imagens
	for (int i = 0; i < self->n_imagens; ++i) {
		self->iLoad.imAtual = i + 1;
		self->targets[i] = Tensor_new(1, self->n_classes, 1, 1, self->cnn->erro, 0, self->cnn->gpu->context, self->cnn->queue);
		self->targets[i]->copyM(self->targets[i], lbgpuREAL, 0, i * self->targets[i]->bytes, self->targets[i]->bytes);
	}
	Release(lbgpuREAL);
	Release(lbgpu);
	self->labels = lbram;
	self->runing = 0;
}


void Setup_treinar(Setup self) {
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
	for (; self->can_run && self->epoca_atual < self->n_epocas && !self->cnn->erro->error; self->epoca_atual++) {
		if (self->imagem_atual_treino >= self->n_imagens_treinar) {
			self->imagem_atual_treino = 0;
			acertos = 0;
		}
		localItrain.epAtual = self->epoca_atual+1;
		for (; self->can_run && self->imagem_atual_treino < self->n_imagens_treinar && !self->cnn->erro->error; self->imagem_atual_treino++) {
			images++;
			indice = self->imagem_atual_treino;
			entrada = self->imagens[indice];
			target = self->targets[indice];
			label = ((char *) self->labels->data)[indice];
			self->cnn->predict(self->cnn, entrada);
			self->cnn->learn(self->cnn, target);
			cnn_label = self->cnn->maxIndex(self->cnn);
			if (self->on_train) { self->on_train(self, label); }

			// #### informações do treinamento
			acertos += (cnn_label == label);
			localItrain.meanwinRate = acertos / (self->imagem_atual_treino + 1.0);
			localItrain.winRate = localItrain.winRate * alpha + beta * ((cnn_label == label) ? 100 : 0);
			localItrain.mse = localItrain.mse * alpha + beta * self->cnn->mse(self->cnn);
			localItrain.imAtual = self->imagem_atual_treino+1;
			self->itrain = localItrain;
			classe++;
		}
	}
	self->runing = 0;
}

void Setup_avaliar(Setup self) {
	// ###  thread de alta prioridade
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	// ### variaveis usadas na avaliação
	Tensor entrada, target;
	ubyte label, cnn_label;
	int indice;
	self->iteste.imTotal = self->n_imagens_testar;
	Iteste localIteste = self->iteste;
	int64_t acertos = 0;
	double t0 = seconds();
	double t1;
	for (; self->can_run && self->imagem_atual_teste < self->n_imagens_testar; self->imagem_atual_teste++) {
		indice = self->n_imagens_treinar + self->imagem_atual_teste;
		entrada = self->imagens[indice];
		target = self->targets[indice];
		label = ((char *) self->labels->data)[indice];

		self->cnn->predict(self->cnn, entrada);
		cnn_label = self->cnn->maxIndex(self->cnn);

		t1 = seconds();
		localIteste.imps = localIteste.imps * alpha + beta / (t1 - t0);
		t0 = t1;
		acertos += (cnn_label == label);
		localIteste.imAtual = self->imagem_atual_teste + 1;
		localIteste.meanwinRate = acertos / (self->imagem_atual_teste + 1.0);
		localIteste.winRate = localIteste.winRate * alpha + beta * ((cnn_label == label) ? 100 : 0);
		localIteste.mse = localIteste.mse * alpha + beta * self->cnn->mseT(self->cnn, target);
		self->iteste = localIteste;

	}
	self->runing = 0;
}

void Setup_getLuaParams(Setup self) {
	ECXPUSH(self->cnn->erro);
	if (!self->cnn) {
		self->cnn->erro->setError(self->cnn->erro, GAB_NULL_POINTER_ERROR);
		return;
	}
	if (self->cnn->erro->error) { return; }
	lua_State *L = self->cnn->LuaVm;
	if (!L) { return; }
	SETUP_GETLUA_INT(self->n_epocas, "Numero_epocas");
	SETUP_GETLUA_STR(self->home, "home");
	SETUP_GETLUA_STR(self->nome, "nome");
	SETUP_GETLUA_INT(self->n_imagens, "Numero_Imagens");
	SETUP_GETLUA_INT(self->n_imagens_treinar, "Numero_ImagensTreino");
	SETUP_GETLUA_INT(self->n_imagens_testar, "Numero_ImagensAvaliacao");
	SETUP_GETLUA_INT(self->n_classes, "Numero_Classes");
	SETUP_GETLUA_STR(self->nome_classes, "classes");
	SETUP_GETLUA_INT(self->header_image, "bytes_remanessentes_imagem");
	SETUP_GETLUA_BOLL(self->use_gpu, "gpu_mem");
	SETUP_GETLUA_INT(self->header_label, "bytes_remanessentes_classes");
	SETUP_GETLUA_STR(self->treino_out, "estatisticasDeTreino");
	SETUP_GETLUA_STR(self->teste_out, "estatiscasDeAvaliacao");
	SETUP_GETLUA_STR(self->file_image, "arquivoContendoImagens");
	SETUP_GETLUA_STR(self->file_label, "arquivoContendoRespostas");
	ECXPOP(self->cnn->erro);
}

void Setup_loadLua(Setup self, const char *lua_file) {
	ECXPUSH(self->cnn->erro);
	if (!lua_file) {
		self->cnn->erro->setError(self->cnn->erro, GAB_NULL_POINTER_ERROR);
		return;
	}
	CnnLuaLoadFile(self->cnn, lua_file);
	Setup_getLuaParams(self);
	if (self->home) {
		SetCurrentDirectoryA(self->home);
	}
	ECXPOP(self->cnn->erro);
}

int Setup_ok(Setup self) {
	return !(self->force_end || self->cnn->erro->error);
}

Setup Setup_new() {
	Setup setup = gab_alloc(sizeof(Setup_t), 1);
	setup->can_run = 1;
	setup->cnn = Cnn_new();
	setup->checkStop = Setup_checkStop;
	setup->loadImagens = Setup_loadImagens;
	setup->loadLabels = Setup_loadLabel;
	setup->treinar = Setup_treinar;
	setup->avaliar = Setup_avaliar;
	setup->release = Setup_release;
	setup->loadLua = Setup_loadLua;
	setup->ok = Setup_ok;
	return setup;
}
