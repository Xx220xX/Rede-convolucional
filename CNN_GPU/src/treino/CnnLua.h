//
// Created by Henrique on 4/10/2021.
//

#ifndef CNN_GPU_CNNLUA_H
#define CNN_GPU_CNNLUA_H

#include "../cnn.h"
#include "../lua/lua.h"
#include "../lua/lualib.h"
#include "../lua/lauxlib.h"
#include "../defaultkernel.h"

Cnn *globalcnn;

int globalLuaError = 0;
#define checkLua(cond, format, ...) if(!(cond)){ \
    luaL_error(L,format,## __VA_ARGS__) ;  \
        globalLuaError = -1;                           \
        return 0;}

#define REGISTERC_L(state, func, name)lua_pushcfunction(state,func);\
lua_setglobal(state,name)

static int l_createCnn(lua_State *L) {
	//printf("l_createCnn\n");
	checkLua(!*globalcnn, "A entrada ja foi definida");
	Params p = {0.1, 0.0, 0.0};
	int x, y, z, d;
	cl_device_type device;
	x = luaL_checkinteger(L, 1);
	y = luaL_checkinteger(L, 2);
	z = luaL_checkinteger(L, 3);
	device = CL_DEVICE_TYPE_GPU;
	if (!lua_isnoneornil(L, 4)) {
		d = luaL_checkinteger(L, 4);
		if (d == 0b10) {
			device = CL_DEVICE_TYPE_CPU;
		}
	}
	*globalcnn = createCnnWithWrapperProgram(default_kernel, p, x, y, z, device);

	return 0;
}

static int l_loadCnn(lua_State *L) {
	//printf("l_loadCnn\n");
	checkLua(*globalcnn, "A entrada não foi definida");
	Params p = {0.1, 0.0, 0.0};
	char *file;
	file = (char *) lua_tostring(L, 1);
	FILE *f = fopen(file, "rb");
	checkLua(f, "arquivo %s nao foi encontrado\n", file);
	cnnCarregar(*globalcnn, f);
	fclose(f);
	return 0;
}

static int l_convolution(lua_State *L) {
	//printf("l_convolution\n");
	int passo, sfiltro, nfiltro;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	passo = luaL_checkinteger(L, 1);
	sfiltro = luaL_checkinteger(L, 2);
	nfiltro = luaL_checkinteger(L, 3);
	int erro = CnnAddConvLayer(*globalcnn, passo, sfiltro, nfiltro);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}

static int l_convolution_non_causal(lua_State *L) {
	//printf("l_convolution_non_causal\n");
	int passox, passoy, largx, largy, filtrox, filtroy, nfiltro;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	passox = luaL_checkinteger(L, 1);
	passoy = luaL_checkinteger(L, 2);
	largx = luaL_checkinteger(L, 3);
	largy = luaL_checkinteger(L, 4);
	filtrox = luaL_checkinteger(L, 5);
	filtroy = luaL_checkinteger(L, 6);
	nfiltro = luaL_checkinteger(L, 7);
	int erro = CnnAddConvNcLayer(*globalcnn, passox, passoy, largx, largy, filtrox, filtroy, nfiltro);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}


static int l_pooling(lua_State *L) {
	//printf("l_pooling\n");
	int passo, sfiltro;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	passo = luaL_checkinteger(L, 1);
	sfiltro = luaL_checkinteger(L, 2);
	int erro = CnnAddPoolLayer(*globalcnn, passo, sfiltro);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}

static int l_poolingav(lua_State *L) {
	//printf("l_poolingav\n");
	int passo, sfiltro;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	passo = luaL_checkinteger(L, 1);
	sfiltro = luaL_checkinteger(L, 2);
	int erro = CnnAddPoolAvLayer(*globalcnn, passo, sfiltro);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}


static int l_relu(lua_State *L) {
	//printf("l_relu\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	int erro = CnnAddReluLayer(*globalcnn);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}

static int l_padding(lua_State *L) {
	//printf("l_padding\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	UINT top = luaL_checkinteger(L, 1);
	UINT bottom = luaL_checkinteger(L, 2);
	UINT left = luaL_checkinteger(L, 3);
	UINT right = luaL_checkinteger(L, 4);
	int erro = CnnAddPaddingLayer(*globalcnn, top, bottom, left, right);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}


static int l_dropout(lua_State *L) {
	//printf("l_dropout\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	double ativa = luaL_checknumber(L, 1);
	long long int seed = time(NULL);
	if (!lua_isnoneornil(L, 2))
		seed = luaL_checkinteger(L, 2);
	int erro = CnnAddDropOutLayer(*globalcnn, ativa, seed);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}

static int l_fullConnect(lua_State *L) {
	//printf("l_fullConnect\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	int neuros = luaL_checkinteger(L, 1);
	int func = luaL_checkinteger(L, 2);
	checkLua(func == FTANH || func == FSIGMOID || func == FRELU, "FUNCAO DE ATIVACAO INVALIDA");
	int erro = CnnAddFullConnectLayer(*globalcnn, neuros, func);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}

static int l_batchnorm(lua_State *L) {
	//printf("l_batchnorm\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	int erro = CnnAddBatchNorm(*globalcnn, 1e-12);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}

static int l_softmax(lua_State *L) {
	//printf("l_batchnorm\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	int erro = CnnAddSoftMax(*globalcnn);
	if (erro) {
		char msg[250];
		getClError(erro, msg);
		luaL_error(L, "falha ao adicionar camada  %s: %d %s", (*globalcnn)->error.context, erro, msg);
	}
	return 0;
}

void loadCnnLuaLibrary(lua_State *L) {
	REGISTERC_L(L, l_createCnn, "Entrada");
	REGISTERC_L(L, l_convolution, "Convolucao");
	REGISTERC_L(L, l_convolution_non_causal, "ConvolucaoNcausal");
	REGISTERC_L(L, l_pooling, "Pooling");
	REGISTERC_L(L, l_poolingav, "PoolingAv");
	REGISTERC_L(L, l_relu, "Relu");
	REGISTERC_L(L, l_padding, "Padding");
	REGISTERC_L(L, l_dropout, "Dropout");
	REGISTERC_L(L, l_fullConnect, "FullConnect");
	REGISTERC_L(L, l_batchnorm, "BatchNorm");
	REGISTERC_L(L, l_softmax, "SoftMax");
	REGISTERC_L(L, l_loadCnn, "CarregarRede");
	lua_pushinteger(L, FSIGMOID);
	lua_setglobal(L, "SIGMOID");
	lua_pushinteger(L, FTANH);
	lua_setglobal(L, "TANH");
	lua_pushinteger(L, FRELU);
	lua_setglobal(L, "RELU");
	lua_pushinteger(L, 0b10);
	lua_setglobal(L, "CPU");
	lua_pushinteger(L, 0b100);
	lua_setglobal(L, "GPU");
}

#define GETLUAVALUE(to, L, key, type, isnil) lua_getglobal(L,key);if(lua_isnoneornil(L,-1)){lua_pop(L,1);isnil}else{to =  luaL_check##type(L,-1);lua_pop(L,1);}

#define GETLUASTRING(to, aux, len, L, key, isnil) lua_getglobal(L,key);if(lua_isnoneornil(L,-1)){lua_pop(L,1);isnil}else{ \
aux = (char *) lua_tostring(L,-1);snprintf(to,len,"%s",aux);lua_pop(L,1);}
#ifndef MAX_STRING_LEN
#define MAX_STRING_LEN 256
#endif
typedef struct {
	char nome[MAX_STRING_LEN];
	char home[MAX_STRING_LEN];
	int Numero_epocas;
	int SalvarBackupACada;
	int Numero_Imagens;
	int Numero_ImagensTreino;
	int Numero_ImagensAvaliacao;
	int Numero_Classes;
	int SalvarSaidasComoPPM;
	int bytes_remanessentes_imagem;
	int bytes_remanessentes_classes;

	char estatisticasDeTreino[MAX_STRING_LEN];
	char estatiscasDeAvaliacao[MAX_STRING_LEN];

	char arquivoContendoImagens[MAX_STRING_LEN];
	char arquivoContendoRespostas[MAX_STRING_LEN];
	Nomes *names;
} ParametrosCnnALL;
#endif //CNN_GPU_CNNLUA_H
