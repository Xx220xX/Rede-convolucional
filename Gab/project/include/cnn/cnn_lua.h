//
// Created by Henrique on 24/11/2021.
//

#ifndef GAB_CNN_CNN_LUA_H
#define GAB_CNN_CNN_LUA_H

#include "cnn/cnn.h"
#include "lua/lua.h"
#include "lua/lualib.h"
#include "lua/lauxlib.h"


extern void CnnInitLuaVm(Cnn self);

extern void loadCnnLuaLibrary(lua_State *L);

extern void enableUtf8();

extern int CnnLuaConsole(Cnn c);


extern int CnnLuaLoadString(Cnn c, const char *lua_program);

extern int CnnLuaLoadFile(Cnn c, const char *file_name);


#endif //GAB_CNN_CNN_LUA_H
