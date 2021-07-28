//
// Created by Henrique on 4/10/2021.
//

#ifndef CNN_GPU_CNNLUA_H
#define CNN_GPU_CNNLUA_H

#include "../../include/cnn/cnn.h"
#include "../lua/lua.h"
#include "../lua/lualib.h"
#include "../lua/lauxlib.h"
#include "../defaultkernel.h"

#ifdef DISABLE_KERNELS_INSIDE_DRIVE
#include "../gpuKernels.h"
#endif
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
#define L_CONVOLUTION_NAME "Convolucao"
static int l_convolution(lua_State *L) {
	int nArgs = lua_gettop(L);
	int px, py, fx, fy, flag = 0;
	int nfiltros;
	int arg = 1;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 3:// px=py, fx=fy,nfilters
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			break;
		case 4://px=py, fx=fy,nfilters, flag
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			flag = luaL_checkinteger(L, arg++);
			break;
		case 5:// px,py, fx,fy,nfilters
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			break;
		case 6:// px,py, fx,fy,,nfilters,flag
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			flag = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n %s(step,filterSize,nfilters)\n"
			              " %s(step,filterSize,nfilters,memory_flag)\n"
			              " %s(stepx,stepy,filterSizex,filterSizey,nfilters)\n"
			              " %s(stepx,stepy,filterSizex,filterSizey,nfilters,memory_flag)\n"
					, L_CONVOLUTION_NAME,L_CONVOLUTION_NAME,L_CONVOLUTION_NAME,L_CONVOLUTION_NAME);
			return 2;
	}
	int erro = Convolucao(*globalcnn, flag, px, py, fx, fy, nfiltros);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}
#define L_CONVOLUTIONNC_NAME "ConvolucaoNcausal"
static int l_convolution_non_causal(lua_State *L) {
	int nArgs = lua_gettop(L);
	int px, py,ax,ay, fx, fy, flag = 0;
	int nfiltros;
	int arg = 1;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 4:// px=py, fx=fy,ax=ay,nfilters
			py = px = luaL_checkinteger(L, arg++);
			ay = ax = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			break;
		case 5://px=py, fx=fy,ax=ay,nfilters, flag
			py = px = luaL_checkinteger(L, arg++);
			ay = ax = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			flag = luaL_checkinteger(L, arg++);
			break;
		case 7:// px,py, fx,fy,ax,ay,nfilters
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			ax = luaL_checkinteger(L, arg++);
			ay = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			break;
		case 8:// px,py, fx,fy,ax,ay,nfilters,flag
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			ax = luaL_checkinteger(L, arg++);
			ay = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			flag = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n %s(step,filterSize,aperture,nfilters)\n"
			              " %s(step,filterSize,aperture,nfilters,memory_flag)\n"
			              " %s(stepx,stepy,filterSizex,filterSizey,aperturex,aperturey,nfilters)\n"
			              " %s(stepx,stepy,filterSizex,filterSizey,aperturex,aperturey,nfilters,memory_flag)\n"
					, L_CONVOLUTIONNC_NAME,L_CONVOLUTIONNC_NAME,L_CONVOLUTIONNC_NAME,L_CONVOLUTIONNC_NAME);
			return 2;
	}
	int erro = ConvolucaoNcausal(*globalcnn, flag, px, py, fx, fy, ax, ay, nfiltros);

	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}

#define L_POOLING_NAME "Pooling"
static int l_pooling(lua_State *L) {
	//printf("l_pooling\n");
	int nArgs = lua_gettop(L);
	int px, py, fx, fy, flag = 0;
	int arg = 1;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 2:// px=py, fx=fy
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			break;
		case 3:// px=py, fx=fy, flag
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			flag = luaL_checkinteger(L, arg++);
			break;
		case 4:// px,py, fx,fy
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			break;
		case 5:// px,py, fx,fy,flag
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n %s(step,filterSize)\n"
			              " %s(step,filterSize,memory_flag)\n"
			              " %s(stepx,stepy,filterSizex,filterSizey)\n"
			              " %s(stepx,stepy,filterSizex,filterSizey,memory_flag)\n"
					, L_POOLING_NAME,L_POOLING_NAME,L_POOLING_NAME,L_POOLING_NAME);
			return 2;
	}
	int erro = Pooling(*globalcnn, flag, px, py, fx, fy);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}
#define L_POOLINGAV_NAME "PoolingAv"
static int l_poolingav(lua_State *L) {
	//printf("l_poolingav\n");
	int nArgs = lua_gettop(L);
	int px, py, fx, fy, flag = 0;
	int arg = 1;
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 2:// px=py, fx=fy
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			break;
		case 3:// px=py, fx=fy, flag
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			flag = luaL_checkinteger(L, arg++);
			break;
		case 4:// px,py, fx,fy
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			break;
		case 5:// px,py, fx,fy,flag
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n %s(step,filterSize)\n"
				 " %s(step,filterSize,memory_flag)\n"
				 " %s(stepx,stepy,filterSizex,filterSizey)\n"
				 " %s(stepx,stepy,filterSizex,filterSizey,memory_flag)\n"
				 , L_POOLINGAV_NAME,L_POOLINGAV_NAME,L_POOLINGAV_NAME,L_POOLINGAV_NAME);
			return 2;
	}
	int erro = PoolingAv(*globalcnn, flag, px, py, fx, fy);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}


static int l_relu(lua_State *L) {
	//printf("l_relu\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	char usehost = 0;
	if (!lua_isnoneornil(L, 1)) {
		usehost = luaL_checkinteger(L, 1);
	}
	int erro = Relu(*globalcnn, usehost);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
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
	char usehost = 0;
	if (!lua_isnoneornil(L, 5)) {
		usehost = luaL_checkinteger(L, 5);
	}
	int erro = Padding(*globalcnn, usehost, top, bottom, left, right);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
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
	char usehost = 0;
	if (!lua_isnoneornil(L, 3)) {
		usehost = luaL_checkinteger(L, 3);
	}
	int erro = Dropout(*globalcnn, usehost, ativa, seed);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}

static int l_fullConnect(lua_State *L) {
	//printf("l_fullConnect\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	int neuros = luaL_checkinteger(L, 1);
	int func = luaL_checkinteger(L, 2);
	checkLua(func == FTANH || func == FSIGMOID || func == FRELU, "FUNCAO DE ATIVACAO INVALIDA");
	char usehost = 0;
	if (!lua_isnoneornil(L, 3)) {
		usehost = luaL_checkinteger(L, 3);
	}
	int erro = FullConnect(*globalcnn, usehost, neuros, func);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}

static int l_batchnorm(lua_State *L) {
	//printf("l_batchnorm\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	char usehost = 0;
	if (!lua_isnoneornil(L, 1)) {
		usehost = luaL_checkinteger(L, 1);
	}
	int erro = BatchNorm(*globalcnn, usehost, 1e-12);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}

static int l_softmax(lua_State *L) {
	//printf("l_batchnorm\n");
	checkLua(*globalcnn, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	char usehost = 0;
	if (!lua_isnoneornil(L, 1)) {
		usehost = luaL_checkinteger(L, 1);
	}
	int erro = SoftMax(*globalcnn, usehost);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	return 0;
}

void loadCnnLuaLibrary(lua_State *L) {
	REGISTERC_L(L, l_createCnn, "Entrada");
	REGISTERC_L(L, l_convolution, L_CONVOLUTION_NAME);
	REGISTERC_L(L, l_convolution_non_causal, L_CONVOLUTIONNC_NAME);
	REGISTERC_L(L, l_pooling, L_POOLING_NAME);
	REGISTERC_L(L, l_poolingav, L_POOLINGAV_NAME);
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

	lua_pushinteger(L, TENSOR_NCPY);
	lua_setglobal(L, "NO_CPY");
	lua_pushinteger(L, TENSOR_SMEM);
	lua_setglobal(L, "SHARED_MEM");


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
