//
// Created by Henrique on 4/10/2021.
//

#ifndef CNN_GPU_CNNLUA_H
#define CNN_GPU_CNNLUA_H

#include <conio.h>
#include <winuser.h>
#include "cnn_lua.h"

#define LCNN "Cnn"

#define checkLua(cond, format, ...) if(!(cond)){ \
    luaL_error(L,format,## __VA_ARGS__) ;  \
        c->erro->error = 40;                           \
        return 0;}

#define REGISTERC_L(state, func, name)lua_pushcfunction(state,func);\
lua_setglobal(state,name)

#define RETURN_LUA_STATUS_FUNCTION() lua_pushinteger(L, c->erro->error);return 1;


typedef struct Luac_function {
	void *f;
	const char *name;
	const char *description;
	const char *sintaxe;
} Luac_function;
typedef struct Luac_contantes {
	uint32_t v;
	const char *name;
} Luac_contantes;
char global_close = 0;

void (*globalFunctionHelpArgs)(void) =NULL;

typedef struct {
	char *str;
	uint32_t len;
	uint32_t n;
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
	while (c->l > 0) {
		c->removeLastLayer(c);
	}
	size_t x, y, z;

	x = luaL_checkinteger(L, 1);
	y = luaL_checkinteger(L, 2);
	z = luaL_checkinteger(L, 3);
	c->setInput(c, x, y, z);
	return 0;
}

/*
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
		c->erro->error = -3 + -100;
		lua_pushinteger(L, c->erro->error);
		char buf[250];
		snprintf(buf, 250, "arquivo %s/%s nao foi encontrado\n", path, file);
		lua_pushstring(L, buf);
		c->erro->error = 0;
		return 2;
	}
	cnnCarregar(c, f);
	fclose(f);
	if (c->erro->error) {
//		luaL_error(L,"%s\n",c->error.msg);
		lua_pushinteger(L, c->erro->error);
		lua_pushstring(L, c->error.msg);
		c->erro->error = 0;
		return 2;
	}
	RETURN_LUA_STATUS_FUNCTION();
}
*/

static int l_Convolucao(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	P2d p = {0};
	P3d f = {0};
	int arg = 1;
	Parametros prm = {0.001};
	RandomParams rdp = {0};
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 3:// px=py, fx=fy,nfilters
			p.x = p.y = luaL_checkinteger(L, arg++);
			f.y = f.x = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			break;
		case 5:// px,py, fx,fy,nfilters
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			break;
		case 8:// px,py, fx,fy,nfilters,typeRand,a,b
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			rdp.type = (int) luaL_checkinteger(L, arg++);
			rdp.a = (REAL) luaL_checknumber(L, arg++);
			rdp.b = (REAL) luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " Convolucao(step,filterSize,nfilters)\n"
						  " Convolucao(stepx,stepy,filterx,filtery,nfilters)\n"
						  " Convolucao(stepx,stepy,filterx,filtery,nfilters,PDF,a,b)\n"
			);

	}
	if (c->Convolucao(c, p, f, prm, rdp)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada Convolucao: %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_ConvolucaoF(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	P2d p = {0};
	P3d f = {0};
	uint32_t fativacao = FTANH;
	int arg = 1;
	Parametros prm = {0.001};
	RandomParams rdp = {0};
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 3:// px=py, fx=fy,nfilters
			p.x = p.y = luaL_checkinteger(L, arg++);
			f.y = f.x = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			break;
		case 4:// px=py, fx=fy,nfilters
			p.x = p.y = luaL_checkinteger(L, arg++);
			f.y = f.x = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			fativacao = luaL_checkinteger(L, arg++);
			break;
		case 5:// px,py, fx,fy,nfilters
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			break;
		case 6:// px,py, fx,fy,nfilters
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			fativacao = luaL_checkinteger(L, arg++);
			break;
		case 9:// px,py, fx,fy,nfilters,typeRand,a,b
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			fativacao = luaL_checkinteger(L, arg++);
			rdp.type = (int) luaL_checkinteger(L, arg++);
			rdp.a = (REAL) luaL_checknumber(L, arg++);
			rdp.b = (REAL) luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " ConvolucaoF(step,filterSize,nfilters)\n"
						  " ConvolucaoF(step,filterSize,nfilters,fativacao)\n"
						  " ConvolucaoF(stepx,stepy,filterx,filtery,nfilters)\n"
						  " ConvolucaoF(stepx,stepy,filterx,filtery,nfilters,fativacao)\n"
						  " ConvolucaoF(stepx,stepy,filterx,filtery,nfilters,fativacao,PDF,a,b)\n"
			);

	}
	if (c->ConvolucaoF(c, p, f, fativacao, prm, rdp)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada ConvolucaoF:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_ConvolucaoNC(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	P2d p = {0};
	P2d a = {0};
	P3d f = {0};
	uint32_t fativacao = FTANH;
	int arg = 1;
	Parametros prm = {0.001};
	RandomParams rdp = {0};
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 4:// px=py,ax=ay, fx=fy,nfilters
			p.x = p.y = luaL_checkinteger(L, arg++);
			a.x = a.y = luaL_checkinteger(L, arg++);
			f.y = f.x = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			break;
		case 5:// px=py,ax=ay, fx=fy,nfilters ,fativacao
			p.x = p.y = luaL_checkinteger(L, arg++);
			a.y = a.x = luaL_checkinteger(L, arg++);
			f.y = f.x = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			fativacao = luaL_checkinteger(L, arg++);
			break;
		case 7:// px,py,ax,ay, fx,fy,nfilters
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			a.x = luaL_checkinteger(L, arg++);
			a.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			break;
		case 8:// px,py,ax,ay, fx,fy,nfilters,fativacao
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			a.x = luaL_checkinteger(L, arg++);
			a.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			fativacao = luaL_checkinteger(L, arg++);
			break;
		case 11:// px,py,ax,ay, fx,fy,nfilters,typeRand,a,b
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			a.x = luaL_checkinteger(L, arg++);
			a.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			f.z = luaL_checkinteger(L, arg++);
			fativacao = luaL_checkinteger(L, arg++);
			rdp.type = (int) luaL_checkinteger(L, arg++);
			rdp.a = (REAL) luaL_checknumber(L, arg++);
			rdp.b = (REAL) luaL_checknumber(L, arg++);

			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " ConvolucaoNC(step,abertura,filterSize,nfilters)\n"
						  " ConvolucaoNC(step,abertura,filterSize,nfilters,fativacao)\n"
						  " ConvolucaoNC(stepx,stepy,aberturax,aberturay,filterx,filtery,nfilters)\n"
						  " ConvolucaoNC(stepx,stepy,aberturax,aberturay,filterx,filtery,nfilters,fativacao)\n"
						  " ConvolucaoNC(stepx,stepy,aberturax,aberturay,filterx,filtery,nfilters,fativacao,PDF,a,b)\n"
			);

	}
	if (c->ConvolucaoNC(c, p, a, f, fativacao, prm, rdp)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada ConvolucaoNC:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_Pooling(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	P2d p = {0};
	P2d f = {0};
	uint32_t type = MAXPOOL;
	int arg = 1;
	Parametros prm = {0.001};
	RandomParams rdp = {0};
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 3:// px=py, fx=fy,nfilters
			type = luaL_checkinteger(L, arg++);
			p.x = p.y = luaL_checkinteger(L, arg++);
			f.y = f.x = luaL_checkinteger(L, arg++);
			break;
		case 5:// px,py, fx,fy,nfilters
			type = luaL_checkinteger(L, arg++);
			p.x = luaL_checkinteger(L, arg++);
			p.y = luaL_checkinteger(L, arg++);
			f.x = luaL_checkinteger(L, arg++);
			f.y = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " Pooling(type,step,filterSize)\n"
						  " Pooling(type,stepx,stepy,filterSizex,filterSizey)\n"

			);

	}
	if (c->Pooling(c, p, f, type)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada Pooling:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}


static int l_Relu(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	REAL less = 0, greater = 1;
	int arg = 1;
	switch (nArgs) {
		case 0:
			break;
		case 2:
			less = (REAL) luaL_checknumber(L, arg++);
			greater = (REAL) luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " Relu()\n"
						  " Relu(lessoh,greateroh)\n"
			);
			return 0;
	}


	if (c->Relu(c, less, greater)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada Relu:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_PRelu(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int arg = 1;
	RdParams rdp = {0};
	switch (nArgs) {
		case 0:
			break;
		case 3:
			rdp.type = (int) luaL_checkinteger(L, arg++);
			rdp.a = (REAL) luaL_checknumber(L, arg++);
			rdp.b = (REAL) luaL_checknumber(L, arg++);
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " PRelu()\n"
						  " PRelu(PDF,a,b)\n"
			);
			return 0;
	}


	if (c->PRelu(c, Params(1e-3), rdp)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada PRelu:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_Padding(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	uint32_t top = luaL_checkinteger(L, 1);
	uint32_t bottom = luaL_checkinteger(L, 2);
	uint32_t left = luaL_checkinteger(L, 3);
	uint32_t right = luaL_checkinteger(L, 4);
	if (nArgs != 4) {
		luaL_error(L, "Invalid function\ntry\n"
					  " Padding(pad_top,pad_bottom,pad_left,pad_right)\n"
		);
		RETURN_LUA_STATUS_FUNCTION();
	}
	if (c->Padding(c, top, bottom, left, right)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada Padding:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_DropOut(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);

	REAL prob;
	cl_long seed;

	switch (nArgs) {
		case 1:
			prob = (REAL) luaL_checknumber(L, 1);
			break;
		case 2:
			prob = (REAL) luaL_checknumber(L, 1);
			seed = luaL_checkinteger(L, 2);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " DropOut(prob_saida)\n"
						  " DropOut(prob_saida,seed)\n");
			return 0;
	}
	if (c->DropOut(c, prob, seed)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada DropOut:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_FullConnect(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int arg = 1;
	int neuros;
	int func = FTANH;
	RdParams rdpw = {0};
	RdParams rdpb = {0};
	switch (nArgs) {
		case 1:
			neuros = (int) luaL_checkinteger(L, arg++);
			break;
		case 2:
			neuros = (int) luaL_checkinteger(L, arg++);
			func = (int) luaL_checkinteger(L, arg++);
			break;
		case 8:
			neuros = (int) luaL_checkinteger(L, arg++);
			func = (int) luaL_checkinteger(L, arg++);
			rdpw.type = (int) luaL_checkinteger(L, arg++);
			rdpw.a = (REAL) luaL_checknumber(L, arg++);
			rdpw.b = (REAL) luaL_checknumber(L, arg++);
			rdpb.type = (int) luaL_checkinteger(L, arg++);
			rdpb.a = (REAL) luaL_checknumber(L, arg++);
			rdpb.b = (REAL) luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " FullConnect(out_size)\n"
						  " FullConnect(out_size,func)\n"
						  " FullConnect(out_size,func,PDF_w,a_w,b_w,PDF_b,a_b,b_b)\n");
			return 0;
	}

	checkLua(func == FTANH || func == FSIGMOID || func == FRELU, "FUNCAO DE ATIVACAO INVALIDA");
	if (c->FullConnect(c, neuros, Params(1e-3), func, rdpw, rdpb)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada FullConnect:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_BatchNorm(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	RdParams randomY = {0};
	RdParams randomB = {0};
	int arg = 1;
	REAL epsilon = (REAL) 1e-12;

	switch (nArgs) {
		case 0:
			break;
		case 1:
			epsilon = (REAL) luaL_checknumber(L, arg++);
			break;
		case 7:
			epsilon = (REAL) luaL_checknumber(L, arg++);
			randomY.type = (int) luaL_checkinteger(L, arg++);
			randomY.a = (REAL) luaL_checknumber(L, arg++);
			randomY.b = (REAL) luaL_checknumber(L, arg++);
			randomB.type = (int) luaL_checkinteger(L, arg++);
			randomB.a = (REAL) luaL_checknumber(L, arg++);
			randomB.b = (REAL) luaL_checknumber(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " BatchNorm()\n"
						  " BatchNorm(epsilon)\n"
						  " BatchNorm(epsilon,PDFY,aY,bY,PDFB,aB,bB)\n");
			return 0;
	}
	if (c->BatchNorm(c, epsilon, Params(1e-3), randomY, randomB)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada DropOut:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_SoftMax(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);

	if (c->SoftMax(c)) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao adicionar camada SoftMax:  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_removeLayer(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	c->removeLastLayer(c);
	RETURN_LUA_STATUS_FUNCTION();
}
void printCNN(Cnn self){
	char *tmp;
	P3d out = self->size_in;
	printf("Entrada(%zu,%zu,%zu)\n", out.x, out.y, out.z);
	for (int i = 0; i < self->l; ++i) {
		tmp = self->cm[i]->getGenerate(self->cm[i]);
		out = self->cm[i]->getOutSize(self->cm[i]);
		printf("%s // [%zu,%zu,%zu]\n", tmp, out.x, out.y, out.z);
		free_mem(tmp);
	}
}
static int l_PrintCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	printCNN(c);
	RETURN_LUA_STATUS_FUNCTION();
}

REAL *getNumbers(lua_State *L, uint32_t *n) {
	lua_settop(L, 1);
	luaL_checktype(L, 1, LUA_TTABLE);
	*n = luaL_len(L, 1);
	REAL *v = alloc_mem(*n, sizeof(REAL));
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
	uint32_t len = 0;
	REAL *input = getNumbers(L, &len);
	c->predictv(c, input);
	free_mem(input);
	if (c->erro->error) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao chamar CnnCall  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_CamadasetParam(lua_State *L) {
	int nArgs = lua_gettop(L);
	REAL ht, mm, dc;
	uint32_t cm;
	int arg = 1;
	int skip_learn = 0;
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	cm = c->l - 1;
	switch (nArgs) {
		case 4:
			cm = lua_tointeger(L, arg++) - 1;
		case 3:
			ht = lua_tonumber(L, arg++);
			mm = lua_tonumber(L, arg++);
			dc = lua_tonumber(L, arg++);
			skip_learn = lua_tonumber(L, arg++);
			break;
		default:
			luaL_error(L, "Expected:\n"
						  "camada:int,hitlearn:float,momento:float,decaimentoPeso:float or \n"
						  "hitlearn:float,momento:float,decaimentoPeso:float\n");

			return 0;
	}

	if (cm < 0 || cm >= c->l) {
		luaL_error(L, "Violação de memória, acesso a posição %d  de %d.\nA posição válida é 1<= i <= %d\n", cm + 1,
				   c->l, c->l);
		return 0;
	}

	c->cm[cm]->params = Params(ht, mm, dc, skip_learn);
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
	if (cm < 0 || cm >= c->l) {
		luaL_error(L, "Violação de memória, acesso a posição %d  de %d.\nA posição válida é 1<= i <= %d\n", cm + 1,
				   c->l, c->l);
		return 0;
	}
	learn = lua_tointeger(L, 2);
	c->cm[cm]->params.skipLearn = !learn;
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_learnCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	uint32_t len = 0;
	REAL *target = getNumbers(L, &len);
	c->learnv(c, target);
	free_mem(target);
	if (c->erro->error) {
		char *msg = c->gpu->errorMsg(c->erro->error);
		luaL_error(L, "falha ao chamar CnnLearn  %d %s", c->erro->error, msg);
		free_mem(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_helpCnn(lua_State *L);



static int l_closeConsole(lua_State *L) {
	global_close = 1;
	return 0;
}

Luac_function globalFunctions[] = {
		{l_createCnn,          "Entrada",         "Cria uma CNN para utilizar",                           "Entrada(x:int,y:int,z:int) "},
		{l_CamadasetLearnable, "SetLearnable",    "Ativa ou desativa a correção de pesos",                "SetLearnable(camada_i:int,learn:boolean)"},
		{l_CamadasetParam,     "SetParams",       "Modifica os parametros da camada",                     "SetParams(camada_i:int,hitlearn:float,momentum:float,decaimento:float"},
		{l_removeLayer,        "RemoveLastLayer", "Remove a ultima camada",                               "RemoveLastLayer()"},
		{l_PrintCnn,           "PrintCnn",        "mostra a rede",                                        "PrintCnn()"},
		{l_callCnn,            "Call",            "Faz a propagação",                                     "Call(input:table)"},
		{l_learnCnn,           "Learn",           "Faz a retro propagação",                               "Learn(target:table)"},
		{l_helpCnn,            "helpCnn",         "Mostra detalhes sobre todas funções",                  "helpCnn()"},
		{l_Convolucao,         "Convolucao",      "Adiciona camada Convolucional",                        "Convolucao(step[x,y],filterSize[x,y],nfilters,randomParanm[pdf,a,bq])"},
		{l_ConvolucaoF,        "ConvolucaoF",     "Adiciona camada Convolucional com função de ativacao", "Convolucao(step[x,y],filterSize[x,y],nfilters,ativacao,randomParanm[pdf,a,bq])"},
		{l_ConvolucaoNC,       "ConvolucaoNC"},
		{l_Pooling,            "Pooling"},
		{l_Relu,               "Relu"},
		{l_PRelu,              "PRelu"},
		{l_Padding,            "Padding"},
		{l_DropOut,            "DropOut"},
		{l_FullConnect,        "FullConnect"},
		{l_BatchNorm,          "BatchNorm"},
		{l_SoftMax,            "SoftMax"},
		{l_closeConsole,       "closeConsole"},
		{NULL,}
};
Luac_contantes globalConstantes[] = {
		{FSIGMOID, "SIGMOID"},
		{FTANH,    "TANH"},
		{FRELU,    "RELU"},
		{FLIN,     "LIN"},
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
	if (!c)return 2;
	if (!c->LuaVm)CnnInitLuaVm(c);

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
		if (!strcmp(cmd.str, "cls")) {
			system("cls");
			continue;
		}
		if (!strcmp(cmd.str, "show")) {
			printCNN(c);
			continue;
		}
		if (!strcmp(cmd.str, "clear")) {
			while (c->l > 0) {
				c->removeLastLayer(c);
			}
			continue;
		}

		if (!strcmp(cmd.str, "help")) {
			luaL_dostring(c->LuaVm, "helpCnn()");
			continue;
		}
		int error = luaL_dostring(c->LuaVm, cmd.str);
		if (error) {
			fflush(stdout);
			fprintf(stderr, "\n Error: %d %d %s\n", lua_gettop(c->LuaVm), error, lua_tostring(c->LuaVm, -1));
			fflush(stderr);

		}
	}
	free_mem(cmd.str);
}

int CnnLuaLoadString(Cnn c, const char *lua_program) {
	if (!c)return 10;
	if (!c->LuaVm)CnnInitLuaVm(c);
	int error = luaL_dostring(c->LuaVm, lua_program);

	if (error) {
		fflush(stdout);
		fprintf(stderr, "\nError: %d %d %s\n", lua_gettop(c->LuaVm), error, lua_tostring(c->LuaVm, -1));
		fflush(stderr);
		c->erro->error = error;
		return error;
	}
	if (c->erro->error) {

		return c->erro->error;
	}

	// retro compatibilidade
	{
		luaL_dostring(c->LuaVm,
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
	if (!c)return 10;
	if (!c->LuaVm)CnnInitLuaVm(c);


	int error = luaL_dofile(c->LuaVm, file_name);

	if (error) {
		fflush(stdout);
		fprintf(stderr, "\nError: %d %d %s\n", lua_gettop(c->LuaVm), error, lua_tostring(c->LuaVm, -1));
		fflush(stderr);
		c->erro->error = error;
		return error;
	}
	if (c->erro->error) {

		return c->erro->error;
	}

	// retro compatibilidade
	{
		luaL_dostring(c->LuaVm,
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




#endif //CNN_GPU_CNNLUA_H
