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
static int l_createCnn(lua_state *L){


	Params p = {0.1, 0.0, 0.0, 1};
	*globalcnn = createCnnWithgpu(KERNEL_FUNCTION_FILE, p, TAMANHO_IMAGEM, TAMANHO_IMAGEM, 1);
}
#endif //CNN_GPU_CNNLUA_H
