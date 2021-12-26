//
// Created by Henrique on 4/10/2021.
//

#ifndef CNN_GPU_CNNLUA_H
#define CNN_GPU_CNNLUA_H

#include <conio.h>
#include <windows.h>
#include <error_list.h>

#include "cnn/cnn_lua.h"

#define LCNN "Cnn"

#define checkLua(cond, format, ...) if(!(cond)){ luaL_error(L,format,##__VA_ARGS__) ;c->ecx->error = GAB_ERRO_LUA;return 0;}

#define REGISTERC_L(state, func, name)lua_pushcfunction(state,func);lua_setglobal(state,name)

#define RETURN_LUA_STATUS_FUNCTION() lua_pushinteger(L, c->ecx->error);return 1;


int loadP3D(lua_State *L, int arg, P3d *p) {

	if (!lua_istable(L, arg)) {
		luaL_error(L, "Esperado um P3D\n");
		return 1;
	}
	lua_getfield(L, arg, "x");
	lua_getfield(L, arg, "y");
	lua_getfield(L, arg, "z");
	p->z = lua_tointeger(L, -1);
	p->y = lua_tointeger(L, -2);
	p->x = lua_tointeger(L, -3);
//	lua_pop(L, -3);
//	lua_pop(L, -2);
//	lua_pop(L, -1);
	return 0;
}

int loadP2D(lua_State *L, int arg, P2d *p) {
	if (!lua_istable(L, arg)) {
		if(lua_isinteger(L,arg)){
			p->x = p->y = luaL_checkinteger(L,arg);
			return 0;
		}
		luaL_error(L, "Esperado um P2D\n");
		return 1;
	}
	lua_getfield(L, arg, "x");
	lua_getfield(L, arg, "y");
	p->y = lua_tointeger(L, -1);
	p->x = lua_tointeger(L, -2);

	return 0;
}

int loadParams(lua_State *L, int arg, Parametros *p) {
	if (!lua_istable(L, arg)) {
		luaL_error(L, "Esperado Params\n");
		return 1;
	}
	lua_getfield(L, arg, "hitlearn");
	lua_getfield(L, arg, "momento");
	lua_getfield(L, arg, "decaimento");
	lua_getfield(L, arg, "skipLearn");
	p->hitlearn = lua_tonumber(L, -4);
	p->momento = lua_tonumber(L, -3);
	p->decaimento = lua_tonumber(L, -2);
	p->skipLearn = lua_tointeger(L, -1);
//	lua_pop(L, -4);
//	lua_pop(L, -3);
//	lua_pop(L, -2);
//	lua_pop(L, -1);
	return 0;
}

int loadRdp(lua_State *L, int arg, RdParams *p) {
	if (!lua_istable(L, arg)) {
		luaL_error(L, "Esperado um RDP\n");
		return 1;
	}
	lua_getfield(L, arg, "type");
	lua_getfield(L, arg, "a");
	lua_getfield(L, arg, "b");
	p->type = lua_tointeger(L, -3);
	p->a = lua_tonumber(L, -2);
	p->b = lua_tonumber(L, -1);

//	lua_pop(L, -3);
//	lua_pop(L, -2);
//	lua_pop(L, -1);
	return 0;
}

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

static int l_sizeout(lua_State *L) {
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	P3d sout = c->getSizeOut(c);
	lua_newtable(L);
	lua_pushinteger(L, sout.x);
	lua_setfield(L, -2, "x");
	lua_pushinteger(L, sout.y);
	lua_setfield(L, -2, "y");
	lua_pushinteger(L, sout.z);
	lua_setfield(L, -2, "z");
	lua_pushinteger(L, sout.x * sout.y * sout.z);
	lua_setfield(L, -2, "length");

	return 1;
}

static int l_Convolucao(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	P2d p = {0};
	P3d f = {0};
	int arg = 1;
	Parametros prm = Params(1e-3);
	RandomParams rdp = {0};
	checkLua(c, "Primeiro informe a entrada com 'entrada(x,y,z)'");
	switch (nArgs) {
		case 2:
			loadP2D(L, arg++, &p);
			loadP3D(L, arg++, &f);
			break;
		case 3:
			loadP2D(L, arg++, &p);
			loadP3D(L, arg++, &f);
			loadParams(L, arg++, &prm);
			break;
		case 4:

			loadP2D(L, arg++, &p);
			loadP3D(L, arg++, &f);
			loadParams(L, arg++, &prm);
			loadRdp(L, arg++, &rdp);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " Convolucao(step,filter)\n"
						  " Convolucao(step,filter,Params)\n"
						  " Convolucao(step,filter,Params,RDP)\n"

					  );
	}
	if (c->Convolucao(c, p, f, prm, rdp)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada Convolucao: %d %s", c->ecx->error, msg);
		gab_free(msg);
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
		case 3:
			loadP2D(L, arg++, &p);
			loadP3D(L, arg++, &f);
			fativacao = lua_tointeger(L, arg++);
			break;
		case 4:
			loadP2D(L, arg++, &p);
			loadP3D(L, arg++, &f);
			fativacao = lua_tointeger(L, arg++);
			loadParams(L, arg++, &prm);
			break;
		case 5:
			loadP2D(L, arg++, &p);
			loadP3D(L, arg++, &f);
			fativacao = lua_tointeger(L, arg++);
			loadParams(L, arg++, &prm);
			loadRdp(L, arg++, &rdp);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " ConvolucaoF(step,filter,ativacao)\n"
						  " ConvolucaoF(step,filter,ativacao,Params)\n"
						  " ConvolucaoF(step,filter,ativacao,Params,RDP)\n");
	}
	checkLua(CHECK_F_ATIVACAO(fativacao), "FUNCAO DE ATIVACAO INVALIDA");

	if (c->ConvolucaoF(c, p, f, fativacao, prm, rdp)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada ConvolucaoF:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
		case 4:
			loadP2D(L, arg++, &p);
			loadP2D(L, arg++, &a);
			loadP3D(L, arg++, &f);
			fativacao = lua_tointeger(L, arg++);
			break;
		case 5:
			loadP2D(L, arg++, &p);
			loadP2D(L, arg++, &a);
			loadP3D(L, arg++, &f);
			fativacao = lua_tointeger(L, arg++);
			loadParams(L, arg++, &prm);
			break;
		case 6:
			loadP2D(L, arg++, &p);
			loadP2D(L, arg++, &a);
			loadP3D(L, arg++, &f);
			fativacao = lua_tointeger(L, arg++);
			loadParams(L, arg++, &prm);
			loadRdp(L, arg++, &rdp);
			break;

		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " ConvolucaoNC(step,abertura,filter,fativacao)\n"
						  " ConvolucaoNC(step,abertura,filter,fativacao,params)\n"
						  " ConvolucaoNC(step,abertura,filter,fativacao,params,RDP)\n"

					  );

	}
	checkLua(CHECK_F_ATIVACAO(fativacao), "FUNCAO DE ATIVACAO INVALIDA");
	if (c->ConvolucaoNC(c, p, a, f, fativacao, prm, rdp)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada ConvolucaoNC:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
			loadP2D(L, arg++, &p);
			loadP2D(L, arg++, &f);
			type = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " Pooling(step,filter,type)\n"

					  );

	}

	if (c->Pooling(c, p, f, type)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada Pooling:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
						  " Relu(lessoh,greateroh)\n");
			return 0;
	}


	if (c->Relu(c, less, greater)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada Relu:  %d %s", c->ecx->error, msg);
		gab_free(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();

}

static int l_PRelu(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int arg = 1;
	RdParams rdp = {0};
	Parametros prm = Params(1e-3);
	switch (nArgs) {
		case 0:
			break;
		case 1:
			loadParams(L, arg++, &prm);
			break;
		case 2:
			loadParams(L, arg++, &prm);
			loadRdp(L, arg++, &rdp);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " PRelu()\n"
						  " PRelu(Params)\n"
						  " PRelu(Params,RDP)\n");
			return 0;
	}


	if (c->PRelu(c, prm, rdp)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada PRelu:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
					  " Padding(pad_top,pad_bottom,pad_left,pad_right)\n");
		RETURN_LUA_STATUS_FUNCTION();
	}
	if (c->Padding(c, top, bottom, left, right)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada Padding:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada DropOut:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
	Parametros prm = Params(1e-3);
	switch (nArgs) {
		case 1:
			neuros = (int) luaL_checkinteger(L, arg++);
			break;
		case 2:
			neuros = (int) luaL_checkinteger(L, arg++);
			func = (int) luaL_checkinteger(L, arg++);
			break;
		case 3:
			neuros = (int) luaL_checkinteger(L, arg++);
			func = (int) luaL_checkinteger(L, arg++);
			loadParams(L, arg++, &prm);
			break;
		case 5:
			neuros = (int) luaL_checkinteger(L, arg++);
			func = (int) luaL_checkinteger(L, arg++);
			loadParams(L, arg++, &prm);
			loadRdp(L, arg++, &rdpw);
			loadRdp(L, arg++, &rdpb);

			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " FullConnect(out_size)\n"
						  " FullConnect(out_size,func)\n"
						  " FullConnect(out_size,func,params)\n"
						  " FullConnect(out_size,func,params,RDPW,RDPB)\n");
			return 0;
	}


	checkLua(CHECK_F_ATIVACAO(func), "FUNCAO DE ATIVACAO INVALIDA");
	if (c->FullConnect(c, neuros, prm, func, rdpw, rdpb)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada FullConnect:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
	size_t batch = 1;
	Parametros prm = Params(1e-3);
	switch (nArgs) {
		case 1:
			batch =  luaL_checkinteger(L, arg++);
			break;
		case 2:
			batch =  luaL_checkinteger(L, arg++);
			epsilon = (REAL) luaL_checknumber(L, arg++);
			break;
		case 3:
			batch =  luaL_checkinteger(L, arg++);
			epsilon = (REAL) luaL_checknumber(L, arg++);
			loadParams(L, arg++, &prm);
			break;
		case 5:
			batch =  luaL_checkinteger(L, arg++);
			epsilon = (REAL) luaL_checknumber(L, arg++);
			loadParams(L, arg++, &prm);
			loadRdp(L, arg++, &randomY);
			loadRdp(L, arg++, &randomB);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " BatchNorm(batch_size)\n"
						  " BatchNorm(batch_size, epsilon)\n"
						  " BatchNorm(batch_size, epsilon,Params)\n"
						  " BatchNorm(batch_size, epsilon,Params,RDPY,RDPB)\n");
			return 0;
	}
	if (c->BatchNorm(c, batch,epsilon, Params(1e-3), randomY, randomB)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada BatchNorm:  %d %s", c->ecx->error, msg);
		gab_free(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_SoftMax(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	int8_t flag = SOFTNORM | SOFTLAST;
	int arg = 1;
	switch (nArgs) {
		case 0:
			break;
		case 1:
			flag = luaL_checkinteger(L, arg++);
			break;
		default:
			luaL_error(L, "Invalid function\ntry\n"
						  " SoftMax()\n"
						  " SoftMax(flag)\n");
			return 0;
	}
	if (c->SoftMax(c, flag)) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao adicionar camada SoftMax:  %d %s", c->ecx->error, msg);
		gab_free(msg);
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

static int l_PrintCnn(lua_State *L) {
	int nArgs = lua_gettop(L);
	lua_getglobal(L, LCNN);
	Cnn c = lua_touserdata(L, -1);
	c->print(c, "--");
	RETURN_LUA_STATUS_FUNCTION();
}

REAL *getNumbers(lua_State *L, uint32_t *n) {
	lua_settop(L, 1);
	luaL_checktype(L, 1, LUA_TTABLE);
	*n = luaL_len(L, 1);
	REAL *v = gab_alloc(*n, sizeof(REAL));
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
	gab_free(input);
	if (c->ecx->error) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao chamar CnnCall  %d %s", c->ecx->error, msg);
		gab_free(msg);
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
		case 5:
			cm = lua_tointeger(L, arg++) - 1;
		case 4:
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
		luaL_error(L, "Violação de memória, acesso a posição %d  de %d.\nA posição válida é 1<= i <= %d\n", cm + 1, c->l, c->l);
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
		luaL_error(L, "Violação de memória, acesso a posição %d  de %d.\nA posição válida é 1<= i <= %d\n", cm + 1, c->l, c->l);
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
	gab_free(target);
	if (c->ecx->error) {
		char *msg = c->gpu->errorMsg(c->ecx->error);
		luaL_error(L, "falha ao chamar CnnLearn  %d %s", c->ecx->error, msg);
		gab_free(msg);
	}
	RETURN_LUA_STATUS_FUNCTION();
}

static int l_helpCnn(lua_State *L);


static struct {
	void *f;
	const char *name;
	const char *args;
} globalFunctions[] = {{.f=l_createCnn, .name="Entrada",         "(x:int, y:int, z:int)"},
					   {l_CamadasetLearnable, "SetLearnable",    "(cm = 1:int, learnable:bool)"},
					   {l_CamadasetParam,     "SetParams",       "(cm=1:int, hitlearn:float, momento:float, decaimento:float, skipLearn:bool )"},
					   {l_removeLayer,        "RemoveLastLayer", "()"},
					   {l_PrintCnn,           "PrintCnn",        ""},
					   {l_callCnn,            "Call",            "(entrada:table)"},
					   {l_learnCnn,           "Learn",           "(entrada:table)"},
					   {l_helpCnn,            "helpCnn",         "()"},
					   {l_sizeout,            "sizeOut",         "()"},
					   {l_Convolucao,         "Convolucao",      "(step:P2D, filter:P3D, params=Params(0):Params,rdp=RDP(0):RDP)"},
					   {l_ConvolucaoF,        "ConvolucaoF",     "(step:P2D, filter:P3D, ativacao:int, params=Params(0):Params, rdp=RDP(0):RDP)"},
					   {l_ConvolucaoNC,       "ConvolucaoNC","(step:P2D, abertura:P2D, filter:P3D, ativacao:int, params=Params(0):Params, rdp=RDP(0):RDP)"},
					   {l_Pooling,            "Pooling","(step:P2D, filter:P2D, type:int)"},
					   {l_Relu,               "Relu","(menorQ0=0:float, maiorQ0=1:float)"},
					   {l_PRelu,              "PRelu","(params=Params(0):Params, rdp=RDP(0):RDP)"},
					   {l_Padding,            "Padding","(top:int, bottom:int, left:int, right:int)"},
					   {l_DropOut,            "DropOut","(prob_saida:float,seed=os.time():int64)"},
					   {l_FullConnect,        "FullConnect","(out_size:int, func:int, params=Params(0):Params, RDPW=RDP(0):RDP, RDPB=RDP(0):RDP)"},
					   {l_BatchNorm,          "BatchNorm","(batch_size, epsilon=1e-12,params=Params(0):Params, RDPY=RDP(0):RDP, RDPB=RDP(0):RDP)"},
					   {l_SoftMax,            "SoftMax","(flag=SOFTNORM|SOFTLAST:int)"},
					   {NULL, NULL}};
static struct {
	uint32_t v;
	const char *name;
} globalConstantes[] = {
						{FSIGMOID,        "SIGMOID"},
						{FTANH,           "TANH"},
						{FRELU,           "RELU"},
						{FLIN,            "LIN"},
						{FALAN,           "ALAN"},
						{SOFTLAST,        "LAST"},
						{SOFTNORM,        "SFNORM"},
						{MAXPOOL,         "MAXPOOL"},
						{MINPOOL,         "MINPOOL"},
						{AVEPOOL,         "AVEPOOL"},
						{TENSOR_GAUSSIAN, "GAUSSIAN"},
						{TENSOR_UNIFORM,  "UNIFORM"},

						{FSIGMOID,        "FSIGMOID"},
						{FTANH,           "FTANH"},
						{FRELU,           "FRELU"},
						{FLIN,            "FLIN"},
						{FALAN,           "FALAN"},
						{SOFTLAST,        "SOFTLAST"},
						{SOFTNORM,        "SOFTNORM"},
						{MAXPOOL,         "MAXPOOL"},
						{MINPOOL,         "MINPOOL"},
						{AVEPOOL,         "AVEPOOL"},
						{TENSOR_GAUSSIAN, "TENSOR_GAUSSIAN"},
						{TENSOR_UNIFORM,  "TENSOR_UNIFORM"},

						{0, NULL}};

void helpCnn(FILE *f, char *pre) {
	fprintf(f, "%sFunctions:\n", pre);
	for (int i = 0; globalFunctions[i].f; i++) {
		fprintf(f, "%s\t%s %s\n", pre, globalFunctions[i].name, globalFunctions[i].args);
	}
	fprintf(f, "%sConstantes:\n", pre);
	for (int i = 0; globalConstantes[i].name; i++) {
		fprintf(f, "%s\t%s\n", pre, globalConstantes[i].name);
	}
}

static int l_helpCnn(lua_State *L) {
	helpCnn(stdout, "");
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


void CnnInitLuaVm(Cnn c) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	loadCnnLuaLibrary(L);
	lua_pushlightuserdata(L, c);
	lua_setglobal(L, LCNN);
	c->LuaVm = L;
	c->releaseL = (void (*)(void *)) lua_close;
	luaL_dostring(c->LuaVm, "function P3D(x, y, z)\n"
							"    if z == nil then\n"
							"        z, y = y, x;\n"
							"    end\n"
							"    return { x = x, y = y, z = z }\n"
							"end\n"
							"function P2D(x, y)\n"
							"    if y == nil then\n"
							"        y = x;\n"
							"    end\n"
							"    return { x = x, y = y }\n"
							"end\n"
							"function Params(hitlearn, momento, decaimento, skip)\n"
							"    momento = momento or 0.0\n"
							"    decaimento = decaimento or 0.0\n"
							"    skip = skip or 0.0\n"
							"    return {\n"
							"        hitlearn = hitlearn,\n"
							"        momento = momento,\n"
							"        decaimento = decaimento,\n"
							"        skipLearn = skip\n"
							"    }\n"
							"end\n"
							"function RDP(type, a, b)\n"
							"    type = type or 0.0\n"
							"    a = a or 0.0\n"
							"    b = b or 0.0\n"
							"    return { type = type, a = a, b = b }\n"
							"end");
}


int CnnLuaLoadString(Cnn c, const char *lua_program) {
	if (!c) { return 10; }
	if (!c->LuaVm) { CnnInitLuaVm(c); }
	int error = luaL_dostring(c->LuaVm, lua_program);
	if (error) {
		fflush(stdout);
		fprintf(stderr, "\nError: %d %d %s\n", lua_gettop(c->LuaVm), error, lua_tostring(c->LuaVm, -1));
		fflush(stderr);
		c->ecx->error = error;
		return error;
	}
	if (c->ecx->error) {

		return c->ecx->error;
	}
}

int CnnLuaLoadFile(Cnn c, const char *file_name) {
	if (!c) { return GAB_NULL_POINTER_ERROR; }
	ECXPUSH(c->ecx);
	if (!c->LuaVm) { CnnInitLuaVm(c); }
	c->ecx->setError(c->ecx, luaL_dofile(c->LuaVm, file_name));
	if (c->ecx->error) {
		fflush(stdout);
		fprintf(stderr, "\nError: %d %d %s\n", lua_gettop(c->LuaVm), c->ecx->error, lua_tostring(c->LuaVm, -1));
		fflush(stderr);
	}
	ECXPOP(c->ecx);
	return c->ecx->error;
}


#endif //CNN_GPU_CNNLUA_H
