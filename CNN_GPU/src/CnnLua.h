//
// Created by Henrique on 4/10/2021.
//

#ifndef CNN_GPU_CNNLUA_H
#define CNN_GPU_CNNLUA_H

#include "cnn.h"
#include "lua/lua.h"
#include "lua/lualib.h"
#include "lua/lauxlib.h"

#include<time.h>

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

static int l_createCnn(lua_State *L) {
	luaL_error(L, "Invalid function\n");
	return 0;
}

static int l_loadCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	Params p = {0.1, 0.0, 0.0};
	char *file;
	file = (char *) lua_tostring(L, 1);
	FILE *f = fopen(file, "rb");
	checkLua(f, "arquivo %s nao foi encontrado\n", file);
	cnnCarregar(c, f);
	fclose(f);
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_convolution(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	printf("%p\n", c);
	int px, py, fx, fy, flag = 0;
	int nfiltros;
	int arg = 1;
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
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
			              " %s(stepx,stepy,filterSizex,filterSizey,nfilters,memory_flag)\n", L_CONVOLUTION_NAME,
			           L_CONVOLUTION_NAME, L_CONVOLUTION_NAME, L_CONVOLUTION_NAME);
			RETURN_LUA_STATUS_FUNCTION();

	}
	c->error.error = Convolucao(c, flag, px, py, fx, fy, nfiltros);
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
			              " %s(stepx,stepy,filterSizex,filterSizey,aperturex,aperturey,nfilters,memory_flag)\n",
			           L_CONVOLUTIONNC_NAME, L_CONVOLUTIONNC_NAME, L_CONVOLUTIONNC_NAME, L_CONVOLUTIONNC_NAME);
			return 0;
	}
	c->error.error = ConvolucaoNcausal(c, flag, px, py, fx, fy, ax, ay, nfiltros);
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
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
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
			              " %s(stepx,stepy,filterSizex,filterSizey,memory_flag)\n", L_POOLING_NAME, L_POOLING_NAME,
			           L_POOLING_NAME, L_POOLING_NAME);
			return 2;
	}
	c->error.error = Pooling(c, flag, px, py, fx, fy);
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
			              " %s(stepx,stepy,filterSizex,filterSizey,memory_flag)\n", L_POOLINGAV_NAME, L_POOLINGAV_NAME,
			           L_POOLINGAV_NAME, L_POOLINGAV_NAME);
			return 2;
	}
	PoolingAv(c, flag, px, py, fx, fy);
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
	char usehost = 0;
	if (!lua_isnoneornil(L, 1)) {
		usehost = luaL_checkinteger(L, 1);
	}
	int erro = Relu(c, usehost);
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
	char usehost = 0;
	if (!lua_isnoneornil(L, 5)) {
		usehost = luaL_checkinteger(L, 5);
	}
	Padding(c, usehost, top, bottom, left, right);
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
	if (!lua_isnoneornil(L, 2))
		seed = luaL_checkinteger(L, 2);
	char usehost = 0;
	if (!lua_isnoneornil(L, 3)) {
		usehost = luaL_checkinteger(L, 3);
	}
	Dropout(c, usehost, ativa, seed);
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
	int neuros = luaL_checkinteger(L, 1);
	int func = luaL_checkinteger(L, 2);
	checkLua(func == FTANH || func == FSIGMOID || func == FRELU, "FUNCAO DE ATIVACAO INVALIDA");
	char usehost = 0;
	if (!lua_isnoneornil(L, 3)) {
		usehost = luaL_checkinteger(L, 3);
	}
	FullConnect(c, usehost, neuros, func);
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
	char usehost = 0;
	if (!lua_isnoneornil(L, 1)) {
		usehost = luaL_checkinteger(L, 1);
	}
	BatchNorm(c, usehost, 1e-12);
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
	char usehost = 0;
	if (!lua_isnoneornil(L, 1)) {
		usehost = luaL_checkinteger(L, 1);
	}
	SoftMax(c, usehost);
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

struct luac_function {
	void *f;
	const char *name;
};
char global_close = 0;

static int l_closeConsole(lua_State *L) {
	global_close = 1;
	return 0;
}

struct luac_function globalFunctions[] = {
		{l_createCnn,    "Entrada"},
		{l_removeLayer,  "RemoveLastLayer"},
		{l_printCnn,     "PrintCnn"},
		{l_callCnn,      "Call"},
		{l_learnCnn,     "Learn"},
		{l_helpCnn,      "helpCnn"},
		{l_convolution,            L_CONVOLUTION_NAME},
		{l_convolution_non_causal, L_CONVOLUTIONNC_NAME},
		{l_pooling,                L_POOLING_NAME},
		{l_poolingav,              L_POOLINGAV_NAME},
		{l_relu,         "Relu"},
		{l_padding,      "Padding"},
		{l_dropout,      "Dropout"},
		{l_fullConnect,  "FullConnect"},
		{l_batchnorm,    "BatchNorm"},
		{l_softmax,      "SoftMax"},
		{l_loadCnn,      "CarregarRede"},
		{l_closeConsole, "closeConsole"},
		{NULL,           ""}
};

static int l_helpCnn(lua_State *L) {
	printf("Funções:\n");
	for (int i = 0; globalFunctions[i].f; i++) {
		printf("%s\n", globalFunctions[i].name);
	}
//	printf("Constantes:\n");
	return 0;
}

void loadCnnLuaLibrary(lua_State *L) {
	for (int i = 0; globalFunctions[i].f; i++) {
		REGISTERC_L(L, globalFunctions[i].f, globalFunctions[i].name);
	}

	lua_pushinteger(L, FSIGMOID);
	lua_setglobal(L, "SIGMOID");
	lua_pushinteger(L, FTANH);
	lua_setglobal(L, "TANH");
	lua_pushinteger(L, FRELU);
	lua_setglobal(L, "RELU");


	lua_pushinteger(L, TENSOR_NCPY);
	lua_setglobal(L, "NO_CPY");
	lua_pushinteger(L, TENSOR_SMEM);
	lua_setglobal(L, "SHARED_MEM");


}


typedef struct {
	char *str;
	UINT len;
	UINT n;
} Comando;
#define INDICADOR ">> "
#define CONTINUA ".. "

#include "windows.h"
#include "conio.h"

int CnnLuaConsole(Cnn c) {
	if (!c)return NULL_PARAM;
	if (!c->L)CnnLoadLua(c);
	Comando cmd = {0};
	int ch = 0;
	global_close = 0;
	cmd.str = alloc_mem(1, 0);
	cmd.len = 1;
	while (!global_close) {
		printf(INDICADOR);
		cmd.n = 0;
		cmd.str[cmd.n] = 0;
		while (1) {
			ch = getche();
			if (ch == '\n' || ch == 13 || ch == '\r') {
				if (GetKeyState(VK_SHIFT) & 0x8000) {
					printf(CONTINUA);
					printf("\n");

				} else {
					printf("\n");
					break;
				}
			}
			if (ch == 8) {
				cmd.n--;
			} else {
				if (cmd.n <= cmd.len - 1) {
					cmd.len++;
					cmd.str = realloc_mem(cmd.str, cmd.len);
				}
				cmd.str[cmd.n] = ch;
				cmd.n++;
			}
			cmd.str[cmd.n] = 0;
		}
		if (!cmd.str[0])continue;
		fflush(stdout);
		int error = luaL_dostring(c->L, cmd.str);
		if (error) {
			fflush(stdout);
			fprintf(stderr, "\n Error: %d %d %s\n", lua_gettop(c->L), error, lua_tostring(c->L, -1));
			fflush(stderr);

		}
	}
	free_mem(cmd.str);
}

#define GETLUAVALUE(to, L, key, type, isnil) lua_getglobal(L,key);if(lua_isnoneornil(L,-1)){lua_pop(L,1);isnil}else{to =  luaL_check##type(L,-1);lua_pop(L,1);}

#define GETLUASTRING(to, aux, len, L, key, isnil) lua_getglobal(L,key);if(lua_isnoneornil(L,-1)){lua_pop(L,1);isnil}else{ \
aux = (char *) lua_tostring(L,-1);snprintf(to,len,"%s",aux);lua_pop(L,1);}

#endif //CNN_GPU_CNNLUA_H
