//
// Created by Henrique on 4/10/2021.
//

#ifndef CNN_GPU_CNNLUA_H
#define CNN_GPU_CNNLUA_H

#include "cnn.h"
#include "lua/lua.h"
#include "lua/lualib.h"
#include "lua/lauxlib.h"
#include "windows.h"
#include "conio.h"
#include<time.h>
#include <utils/dir.h>

#define LCNN "Cnn"
#define checkLua(cond, format, ...) if(!(cond)){ \
    luaL_error(L,format,## __VA_ARGS__) ;  \
        c->error.error = CNN_LUA_ERROR;                           \
        return 0;}

#define REGISTERC_L(state, func, name)lua_pushcfunction(state,func);\
lua_setglobal(state,name)

#define RETURN_LUA_STATUS_FUNCTION() lua_pushinteger(L, c->error.error);return 1;
#define L_CONVOLUTION_NAME "Convolucao"

#define L_CONVOLUTIONNC_NAME "ConvolucaoNcausal"

#define L_POOLING_NAME "Pooling"

#define L_POOLINGAV_NAME "PoolingAv"


typedef struct Luac_function {
	void *f;
	const char *name;
	const char *description;
	const char *sintaxe;
} Luac_function;
typedef struct Luac_contantes {
	UINT v;
	const char *name;
} Luac_contantes;
char global_close = 0;

void (*globalFunctionHelpArgs)(void) =NULL;

typedef struct {
	char *str;
	UINT len;
	UINT n;
} Comando;

#define INDICADOR ">> "
#define CONTINUA ".. "


static int l_createCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	if (nArgs != 3) {
		luaL_error(L, "Expected x:int, y:int, z:int");
		return 0;
	}
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	size_t x, y, z;
	x = luaL_checkinteger(L, 1);
	y = luaL_checkinteger(L, 2);
	z = luaL_checkinteger(L, 3);
	static int count = 0;
	count++;
	if (c->size != 0) {

		luaL_error(L, "O tamanho da rede não pode ser alterado (%d,%d)",count,c->size);
		return 0;
	}

	c->sizeIn = (Ponto) {x, y, z};
	return 0;
}

static int l_loadCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	Params p = {0.1, 0.0, 0.0};
	char *file;
	file = (char *) lua_tostring(L, 1);
	lua_getglobal(L, "home");
	if (lua_isnoneornil(L, -1)) {
		luaL_error(L, "diretorio home não foi definido\n", file);
		return 0;
	}
	const char *path = lua_tostring(L, -1);

	SetDir((char *) path);
	FILE *f = fopen(file, "rb");
	if (!(f)) {
		c->error.error = -3 + -100;
		lua_pushinteger(L, c->error.error);
		char buf[250];
		snprintf(buf, 250, "arquivo %s/%s nao foi encontrado\n", path, file);
		lua_pushstring(L, buf);
		c->error.error = 0;
		return 2;
	}
	cnnCarregar(c, f);
	fclose(f);
	if (c->error.error) {
//		luaL_error(L,"%s\n",c->error.msg);
		lua_pushinteger(L, c->error.error);
		lua_pushstring(L, c->error.msg);
		c->error.error = 0;
		return 2;
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_convolution(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int px, py, fx, fy;
	int nfiltros;
	int arg = 1;
	RandomParam randomParam = {0};
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 3:// px=py, fx=fy,nfilters
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			break;
		case 5:// px,py, fx,fy,nfilters
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			break;
		case 8:// px,py, fx,fy,nfilters,typeRand,a,m
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			randomParam.type = luaL_checkinteger(L, arg++);
			randomParam.a = luaL_checknumber(L, arg++);
			randomParam.b = luaL_checknumber(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n %s(step,filterSize,nfilters)\n"
						  " "L_CONVOLUTION_NAME"(step,filterSize,nfilters)\n"
						  " "L_CONVOLUTION_NAME"(stepx,stepy,filterx,filtery,nfilters)\n"
						  " "L_CONVOLUTION_NAME"(stepx,stepy,filterx,filtery,nfilters,PDF,a,b)\n"
			);

	}
	c->error.error = Convolucao(c, px, py, fx, fy, nfiltros, randomParam);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_convolution_non_causal(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int px, py, ax, ay, fx, fy, flag = 0;
	int nfiltros;
	int arg = 1;
	RandomParam randomParam = {0};
	switch (nArgs) {
		case 4:// px=py, fx=fy,ax=ay,nfilters
			py = px = luaL_checkinteger(L, arg++);
			ay = ax = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
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
		case 10:// px,py, fx,fy,ax,ay,nfilters
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			ax = luaL_checkinteger(L, arg++);
			ay = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			nfiltros = luaL_checkinteger(L, arg++);
			randomParam.type = luaL_checkinteger(L, arg++);
			randomParam.a = luaL_checknumber(L, arg++);
			randomParam.b = luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " "L_CONVOLUTIONNC_NAME"(step,filterSize,aperture,nfilters,memory_flag)\n"
						  " "L_CONVOLUTIONNC_NAME"(stepx,stepy,filterx,filtery,aperturex,aperturey,nfilters)\n"
						  " "L_CONVOLUTIONNC_NAME"(stepx,stepy,filterx,filtery,aperturex,aperturey,nfilters,PDF,a,b)\n"
			);
			return 0;
	}
	c->error.error = ConvolucaoNcausal(c, px, py, fx, fy, ax, ay, nfiltros, randomParam);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_pooling(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int px, py, fx, fy, flag = 0;
	int arg = 1;
	RandomParam randomParam = {0};
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 2:// px=py, fx=fy
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			break;
		case 4:// px,py, fx,fy
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			break;
		case 7:// px,py, fx,fy
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			randomParam.type = luaL_checkinteger(L, arg++);
			randomParam.a = luaL_checknumber(L, arg++);
			randomParam.b = luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " "L_POOLING_NAME"(step,filterSize,memory_flag)\n"
						  " "L_POOLING_NAME"(stepx,stepy,filtrox,filtroy)\n"
						  " "L_POOLING_NAME"(stepx,stepy,filtrox,filtroy,PDF,a,b)\n"
			);
			return 2;
	}
	c->error.error = Pooling(c, px, py, fx, fy);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_poolingav(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int px, py, fx, fy, flag = 0;
	int arg = 1;
	switch (nArgs) {
		case 2:// px=py, fx=fy
			py = px = luaL_checkinteger(L, arg++);
			fy = fx = luaL_checkinteger(L, arg++);
			break;
		case 4:// px,py, fx,fy
			px = luaL_checkinteger(L, arg++);
			py = luaL_checkinteger(L, arg++);
			fx = luaL_checkinteger(L, arg++);
			fy = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " "L_POOLINGAV_NAME"(step,filter)\n"
						  " "L_POOLINGAV_NAME"(stepx,stepy,filterx,filtery)\n"
			);
			return 0;
	}
	PoolingAv(c, px, py, fx, fy);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_relu(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int arg = 1;
	switch (nArgs) {
		case 0:
			break;

		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " Relu()\n"
			);
			return 0;
	}

	int erro = Relu(c);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_prelu(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int arg = 1;
	RandomParam randomParam = {0};
	switch (nArgs) {
		case 0:
			break;
		case 3:
			randomParam.type = luaL_checkinteger(L, arg++);
			randomParam.a = luaL_checknumber(L, arg++);
			randomParam.b = luaL_checknumber(L, arg++);
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " PRelu()\n"
						  " PRelu(PDF,a,b)\n"
			);
			return 0;
	}

	int erro = PRelu(c, randomParam);
	if (erro) {
		char msg[EXCEPTION_MAX_MSG_SIZE];
		getClError(erro, msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", erro, msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_padding(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	UINT top = luaL_checkinteger(L, 1);
	UINT bottom = luaL_checkinteger(L, 2);
	UINT left = luaL_checkinteger(L, 3);
	UINT right = luaL_checkinteger(L, 4);
	if (nArgs != 4) {
		luaL_error(L, "Invalid function\ntry\n"
					  " Padding(pad_top,pad_bottom,pad_left,pad_right)\n"
		);
		return 0;
	}
	Padding(c, top, bottom, left, right);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_dropout(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	double ativa = luaL_checknumber(L, 1);
	long long int seed = time(NULL);
	switch (nArgs) {
		case 1:
			break;
		case 2:
			seed = luaL_checkinteger(L, 2);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " Dropout(prob_saida)\n"
						  " Dropout(prob_saida,seed)\n");
			return 0;
	}

	Dropout(c, ativa, seed);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_fullConnect(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int arg = 1;
	int neuros;
	int func = FTANH;
	RandomParam randomParam = {0};
	switch (nArgs) {
		case 1:
			neuros = luaL_checkinteger(L, arg++);
			break;
		case 2:
			neuros = luaL_checkinteger(L, arg++);
			func = luaL_checkinteger(L, arg++);
			break;
		case 5:
			neuros = luaL_checkinteger(L, arg++);
			func = luaL_checkinteger(L, arg++);
			randomParam.type = luaL_checkinteger(L, arg++);
			randomParam.a = luaL_checknumber(L, arg++);
			randomParam.b = luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " FullConnect(out_size)\n"
						  " FullConnect(out_size,func)\n"
						  " FullConnect(out_size,func,PDF,a,b)\n");
			return 0;
	}

	checkLua(func == FTANH || func == FSIGMOID || func == FRELU, "FUNCAO DE ATIVACAO INVALIDA");
	FullConnect(c, neuros, func, randomParam);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_batchnorm(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	RandomParam randomY = {0};
	RandomParam randomB = {0};
	int arg = 1;
	double epsilon = 1e-12;

	switch (nArgs) {
		case 0:
			break;
		case 1:
			epsilon = luaL_checknumber(L, arg++);
			break;
		case 4:
			epsilon = luaL_checknumber(L, arg++);
			randomY.type = luaL_checkinteger(L, arg++);
			randomY.a = luaL_checknumber(L, arg++);
			randomY.b = luaL_checknumber(L, arg++);
			break;
		case 7:
			epsilon = luaL_checknumber(L, arg++);
			randomY.type = luaL_checkinteger(L, arg++);
			randomY.a = luaL_checknumber(L, arg++);
			randomY.b = luaL_checknumber(L, arg++);
			randomB.type = luaL_checkinteger(L, arg++);
			randomB.a = luaL_checknumber(L, arg++);
			randomB.b = luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " BatchNorm()\n"
						  " BatchNorm(epsilon)\n"
						  " BatchNorm(epsilon,PDFY,aY,bY,PDFB,aB,bB)\n");
			return 0;
	}
	BatchNorm(c, epsilon, randomY, randomB);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
	return 0;
}

static int l_softmax(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	SoftMax(c);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_removeLayer(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	CnnRemoveLastLayer(c);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_printCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	printCnn(c);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao adicionar camada  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

double *getNumbers(lua_State *L, UINT *n) {
	lua_settop(L, 1);
	luaL_checktype(L, 1, LUA_TTABLE);
	*n = luaL_len(L, 1);
	double *v = alloc_mem(*n, sizeof(double));
	for (int i = 0; i < *n; i++) {
		lua_rawgeti(L, 1, i + 1);
		v[i] = lua_tonumber(L, -1);
	}
	return v;
}

static int l_callCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	UINT len = 0;
	double *input = getNumbers(L, &len);
	CnnCall(c, input);
	free_mem(input);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao chamar CnnCall  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_putlua_arg(lua_State *L) {
	if (lua_isnoneornil(L, 2))return 0;
	int nArgs = lua_gettop(L);
	if (nArgs != 2) {
		luaL_error(L, "Expected name:str,value:str\n");
		return 0;
	}
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);

	Dict_push(&c->luaArgs, lua_tostring(L, 1), lua_tostring(L, 2));
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_CamadasetParam(lua_State *L) {
	int nArgs = lua_gettop(L);
	double ht, mm, dc;
	int cm ;
	int arg = 1;
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	cm = c->size-1;
	switch (nArgs) {
		case 4:
			cm = lua_tointeger(L, arg++) - 1;
		case 3:
			ht = lua_tonumber(L, arg++);
			mm = lua_tonumber(L, arg++);
			dc = lua_tonumber(L, arg++);
			break;
		default:
			luaL_error(L, "Expected camada:int,hitlearn:float,momento:float,decaimentoPeso:float or \n"
						  "hitlearn:float,momento:float,decaimentoPeso:float");

			return 0;
	}

	if (cm < 0 || cm >= c->size) {
		luaL_error(L, "Violação de memória, acesso a posição %d  de %d.\nA posição válida é 1<= i <= %d\n", cm + 1,
				   c->size, c->size);
		return 0;
	}

	c->camadas[cm]->setParams(c->camadas[cm], ht, mm, dc);
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_CamadasetLearnable(lua_State *L) {
	int nArgs = lua_gettop(L);
	if (nArgs != 2) {
		luaL_error(L, "Expected camada:int,learnable:boolean\n");
		return 0;
	}
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int learn;
	int cm = lua_tointeger(L, 1) - 1;
	if (cm < 0 || cm >= c->size) {
		luaL_error(L, "Violação de memória, acesso a posição %d  de %d.\nA posição válida é 1<= i <= %d\n", cm + 1,
				   c->size, c->size);
		return 0;
	}
	learn = lua_tointeger(L, 2);
	c->camadas[cm]->setLearn(c->camadas[cm], learn);
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_learnCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	UINT len = 0;
	double *target = getNumbers(L, &len);
	CnnLearn(c, target);
	free_mem(target);
	if (c->error.error) {
		getClError(c->error.error, c->error.msg, EXCEPTION_MAX_MSG_SIZE);
		luaL_error(L, "falha ao chamar CnnLearn  %d %s", c->error.error, c->error.msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_helpCnn(lua_State *L);


static int l_closeConsole(lua_State *L) {
	global_close = 1;
	return 0;
}

Luac_function globalFunctions[] = {
		{l_createCnn,          "Entrada",              "Cria uma CNN para utilizar",            "Entrada(x:int,y:int,z:int) "},
		{l_CamadasetLearnable, "SetLearnable",         "Ativa ou desativa a correção de pesos", "SetLearnable(camada_i:int,learn:boolean)"},
		{l_CamadasetParam,     "SetParams",            "Modifica os parametros da camada",      "SetParams(camada_i:int,hitlearn:float,momentum:float,decaimento:float"},
		{l_removeLayer,        "RemoveLastLayer",      "Remove a ultima camada",                "RemoveLastLayer()"},
		{l_printCnn,           "PrintCnn",             "mostra a rede",                         "PrintCnn()"},
		{l_putlua_arg,         "Args",                 "Utilizado para comunicar com managetrain informando parametros para o treinamento"},
		{l_callCnn,            "Call",                 "Faz a propagação",                      "Call(input:table)"},
		{l_learnCnn,           "Learn",                "Faz a retro propagação",                "Learn(target:table)"},
		{l_helpCnn,            "helpCnn",              "Mostra detalhes sobre todas funções",   "helpCnn()"},
		{l_convolution,            L_CONVOLUTION_NAME, "Adiciona camada Convolucional", L_CONVOLUTION_NAME"(step[x,y],filterSize[x,y],nfilters,randomParanm[pdf,a,bq])"},
		{l_convolution_non_causal, L_CONVOLUTIONNC_NAME},
		{l_pooling,                L_POOLING_NAME},
		{l_poolingav,              L_POOLINGAV_NAME},
		{l_relu,               "Relu"},
		{l_prelu,              "PRelu"},
		{l_padding,            "Padding"},
		{l_dropout,            "Dropout"},
		{l_fullConnect,        "FullConnect"},
		{l_batchnorm,          "BatchNorm"},
		{l_softmax,            "SoftMax"},
		{l_loadCnn,            "CarregarRede"},
		{l_closeConsole,       "closeConsole"},
		{NULL,                 ""}
};
Luac_contantes globalConstantes[] = {
		{FSIGMOID, "SIGMOID"},
		{FTANH,    "TANH"},
		{FRELU,    "RELU"},
		{0, NULL}
};

static int l_helpCnn(lua_State *L) {
	printf("Functions:\n");
	for (int i = 0; globalFunctions[i].f; i++) {
		printf("\t%s\n", globalFunctions[i].name);
		if (globalFunctions[i].description) {
			printf("\t\t%s\n", globalFunctions[i].description);
		}
		if (globalFunctions[i].sintaxe) {
			printf("\t\t%s\n", globalFunctions[i].sintaxe);
		}
	}
	printf("Constantes:\n");
	for (int i = 0; globalConstantes[i].name; i++) {
		printf("\t%s\n", globalConstantes[i].name);
	}
//	if (globalFunctionHelpArgs)
//		globalFunctionHelpArgs();
	return 0;
}

void loadCnnLuaLibrary(lua_State *L) {
	for (int i = 0; globalFunctions[i].f; i++) {
		REGISTERC_L(L, globalFunctions[i].f, globalFunctions[i].name);
	}
	for (int i = 0; globalConstantes[i].name; i++) {
		lua_pushinteger(L, globalConstantes[i].v);
		lua_setglobal(L, globalConstantes[i].name);
	}
}

void LuaputHelpFunctionArgs(void (*myf)()) {
	globalFunctionHelpArgs = myf;
}

int CnnLuaConsole(Cnn c) {
	if (!c)return NULL_PARAM;
	if (!c->L)CnnInitLuaVm(c);

	Comando cmd = {0};
	int ch = 0;
	global_close = 0;
	cmd.str = alloc_mem(1, 0);
	cmd.len = 1;
	printf("console commands:\ncls : clear screen\nclear : remove all layers\nshow : show cnn\nexit : close console\nhelp : call helpCnn()\n");
	while (!global_close) {
		printf(INDICADOR);
		cmd.n = 0;
		cmd.str[cmd.n] = 0;
		while (1) {
			ch = getch();
			if (ch == '\n' || ch == '\r') {
				if (GetKeyState(VK_SHIFT) & 0x8000) {
					printf(CONTINUA);
					printf("\n");

				} else {
					printf("\n");
					break;
				}
			}
			if (ch == 8) {
				if (cmd.n >= 1) {
					cmd.n--;
					printf("\b");
					printf(" ");
					printf("\b");

				}
			} else {
				if (cmd.n <= cmd.len - 1) {
					cmd.len++;
					cmd.str = realloc_mem(cmd.str, cmd.len);
				}
				cmd.str[cmd.n] = ch;
				cmd.n++;
				printf("%c", ch);
			}
			cmd.str[cmd.n] = 0;
		}
		if (!cmd.str[0])continue;
		fflush(stdout);
		if (!strcmp(cmd.str, "exit"))break;
		if (!strcmp(cmd.str, "cls")){
			system("cls");
			continue;
		}
		if (!strcmp(cmd.str, "show")){
			printCnn(c);
			continue;
		}
		if (!strcmp(cmd.str, "clear")){
			while(c->size>0){
				CnnRemoveLastLayer(c);
			}
			continue;
		}

		if (!strcmp(cmd.str, "help")) {
			luaL_dostring(c->L, "helpCnn()");
			continue;
		}
		int error = luaL_dostring(c->L, cmd.str);
		if (error) {
			fflush(stdout);
			fprintf(stderr, "\n Error: %d %d %s\n", lua_gettop(c->L), error, lua_tostring(c->L, -1));
			fflush(stderr);

		}
	}
	free_mem(cmd.str);
}
int CnnLuaLoadString(Cnn c, const char *lua_program){
	if (!c)return NULL_PARAM;
	if (!c->L)CnnInitLuaVm(c);
	int error = luaL_dostring(c->L, lua_program);

	if (error) {
		fflush(stdout);
		fprintf(stderr, "\nError: %d %d %s\n", lua_gettop(c->L), error, lua_tostring(c->L, -1));
		fflush(stderr);
		c->error.error = error;
		return error;
	}
	if (c->error.error) {

		return c->error.error;
	}

	// retro compatibilidade
	{
		luaL_dostring(c->L,
					  "Args('work_path', home)\n"
					  "Args('file_image', arquivoContendoImagens)\n"
					  "Args('file_label', arquivoContendoRespostas)\n"
					  "Args('header_image', bytes_remanessentes_imagem)\n"
					  "Args('header_label', bytes_remanessentes_classes)\n"
					  "Args('numero_epocas', Numero_epocas)\n"
					  "Args('numero_imagens', Numero_Imagens)\n"
					  "Args('numero_treino', Numero_ImagensTreino)\n"
					  "Args('numero_fitnes', Numero_ImagensAvaliacao)\n"
					  "Args('numero_classes', Numero_Classes)\n"
					  "Args('sep', 32)\n"
					  "local nome_classes_lua\n"
					  "local sep_lua\n"
					  "sep_lua = ' '\n"
					  "if sep ~= nil then\n"
					  "    sep_lua = sep\n"
					  "end\n"
					  "for _, v in pairs(classes) do\n"
					  "    if nome_classes_lua == nil then\n"
					  "        nome_classes_lua = v\n"
					  "    else\n"
					  "        nome_classes_lua = nome_classes_lua .. sep_lua .. v\n"
					  "    end\n"
					  "end\n"
					  "Args('nome_classes', nome_classes_lua)");
	}
}
int CnnLuaLoadFile(Cnn c, const char *file_name) {
	if (!c)return NULL_PARAM;
	if (!c->L)CnnInitLuaVm(c);


	int error = luaL_dofile(c->L, file_name);

	if (error) {
		fflush(stdout);
		fprintf(stderr, "\nError: %d %d %s\n", lua_gettop(c->L), error, lua_tostring(c->L, -1));
		fflush(stderr);
		c->error.error = error;
		return error;
	}
	if (c->error.error) {

		return c->error.error;
	}

	// retro compatibilidade
	{
		luaL_dostring(c->L,
					  "Args('work_path', home)\n"
					  "Args('file_image', arquivoContendoImagens)\n"
					  "Args('file_label', arquivoContendoRespostas)\n"
					  "Args('header_image', bytes_remanessentes_imagem)\n"
					  "Args('header_label', bytes_remanessentes_classes)\n"
					  "Args('numero_epocas', Numero_epocas)\n"
					  "Args('numero_imagens', Numero_Imagens)\n"
					  "Args('numero_treino', Numero_ImagensTreino)\n"
					  "Args('numero_fitnes', Numero_ImagensAvaliacao)\n"
					  "Args('numero_classes', Numero_Classes)\n"
					  "Args('sep', 32)\n"
					  "local nome_classes_lua\n"
					  "local sep_lua\n"
					  "sep_lua = ' '\n"
					  "if sep ~= nil then\n"
					  "    sep_lua = sep\n"
					  "end\n"
					  "for _, v in pairs(classes) do\n"
					  "    if nome_classes_lua == nil then\n"
					  "        nome_classes_lua = v\n"
					  "    else\n"
					  "        nome_classes_lua = nome_classes_lua .. sep_lua .. v\n"
					  "    end\n"
					  "end\n"
					  "Args('nome_classes', nome_classes_lua)");
	}
}

int CnnCallT(Cnn c, Tensor input) {
	if (!c)return NULL_PARAM;
	if (!input)return NULL_PARAM;
	if (!c->size)return NULL_PARAM;
	int erro = 0;
	Tensor aux;
	switch (input->flag & TENSOR_MASK_MEM) {
		case TENSOR_SVM:
		case TENSOR_RAM:
			return CnnCall(c, input->hostd);
		case TENSOR_GPU:
			aux = c->camadas[0]->entrada;
			c->camadas[0]->entrada = input;
			erro = CnnCall(c, NULL);
			c->camadas[0]->entrada = aux;
			return erro;
		default:
			return TENSOR_INVALID_FLAG_MEM;
	}
}

int CnnLearnT(Cnn c, Tensor target) {
	if (!c)return NULL_PARAM;
	if (!target)return NULL_PARAM;
	if (!c->size)return NULL_PARAM;
	int erro = 0;
	Tensor aux;
	switch (target->flag & TENSOR_MASK_MEM) {
		case TENSOR_SVM:
		case TENSOR_RAM:
			return CnnLearn(c, target->hostd);
		case TENSOR_GPU:
			aux = c->target;
			c->target = target;
			erro = CnnLearn(c, NULL);
			c->target = aux;
			return erro;
		default:
			return TENSOR_INVALID_FLAG_MEM;
	}
}

#endif //CNN_GPU_CNNLUA_H
