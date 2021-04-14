//
// Created by Henrique on 4/10/2021.
//

#ifndef CNN_GPU_CNNLUA_H
#define CNN_GPU_CNNLUA_H

#include "cnn.h"
#include "lua/lua.h"
#include "lua/lualib.h"
#include "lua/lauxlib.h"

Cnn *globalcnn;
char *globalKernel;
int globalLuaError = 0;
#define checkLua(cond, format, ...) if(!(cond)){ \
    luaL_error(L,format,## __VA_ARGS__) ;  \
        globalLuaError = -1;                           \
        return 0;}

#define REGISTERC_L(state, func, name)lua_pushcfunction(state,func);\
lua_setglobal(state,name)

static int l_createCnn(lua_State *L) {
	checkLua(!*globalcnn,"A entrada ja foi definida");
	Params p = {0.1, 0.0, 0.0, 1};
	int x, y, z;
	x = luaL_checkinteger(L, 1);
	y = luaL_checkinteger(L, 2);
	z = luaL_checkinteger(L, 3);
	*globalcnn = createCnnWithgpu(globalKernel, p, x, y, z);
	return 0;
}

static int l_loadCnn(lua_State *L) {
	checkLua(*globalcnn,"A entrada não foi definida");
	Params p = {0.1, 0.0, 0.0, 1};
	char *file ;
	file = (char *)lua_tostring(L, 1);
	FILE *f = fopen(file,"rb");
	checkLua(f,"arquivo %s nao foi encontrado\n",file);
	cnnCarregar(*globalcnn ,f);
	fclose(f);
	return 0;
}
static int l_convolution(lua_State *L) {
	int passo, sfiltro, nfiltro;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	passo = luaL_checkinteger(L, 1);
	sfiltro = luaL_checkinteger(L, 2);
	nfiltro = luaL_checkinteger(L, 3);
	CnnAddConvLayer(*globalcnn, passo, sfiltro, nfiltro);

	return 0;
}


static int l_pooling(lua_State *L) {
	int passo, sfiltro;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	passo = luaL_checkinteger(L, 1);
	sfiltro = luaL_checkinteger(L, 2);
	CnnAddPoolLayer(*globalcnn, passo, sfiltro);
	return 0;
}


static int l_relu(lua_State *L) {
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	CnnAddReluLayer(*globalcnn);
	return 0;
}


static int l_dropout(lua_State *L) {
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	double ativa = luaL_checknumber(L, 1);
	long long int seed = luaL_checkinteger(L, 2);
	CnnAddDropOutLayer(*globalcnn, ativa, seed);
	return 0;
}

static int l_fullConnect(lua_State *L) {
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	int neuros = luaL_checkinteger(L, 1);
	int func = luaL_checkinteger(L, 2);
	CnnAddFullConnectLayer(*globalcnn, neuros, func);
	return 0;
}

void loadCnnLuaLibrary(lua_State *L) {
	REGISTERC_L(L, l_createCnn, "Entrada");
	REGISTERC_L(L, l_convolution, "Convolucao");
	REGISTERC_L(L, l_pooling, "Pooling");
	REGISTERC_L(L, l_relu, "Relu");
	REGISTERC_L(L, l_dropout, "Dropout");
	REGISTERC_L(L, l_fullConnect, "FullConnect");
	REGISTERC_L(L, l_loadCnn, "CarregarRede");
	lua_pushinteger(L, FSIGMOID);
	lua_setglobal(L, "SIGMOID");
	lua_pushinteger(L, FTANH);
	lua_setglobal(L, "TANH");
	lua_pushinteger(L, FRELU);
	lua_setglobal(L, "RELU");
}

#define GETLUAVALUE(to, L, key, type, isnil) lua_getglobal(L,key);if(lua_isnoneornil(L,-1)){lua_pop(L,1);isnil}else{to =  luaL_check##type(L,-1);lua_pop(L,1);}

#define GETLUASTRING(to,aux,len, L, key,  isnil) lua_getglobal(L,key);if(lua_isnoneornil(L,-1)){lua_pop(L,1);isnil}else{ \
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
