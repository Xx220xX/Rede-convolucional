//
// Created by Henrique on 25/11/2021.
//
#include <mem.h>
#include <windows.h>
#include "cnn/cnn_lua.h"
#include "wchar.h"
#include "conio.h"
#include "lua/console.h"

typedef struct {
	char *str;
	uint32_t len;
	uint32_t n;
} Comando;

#define INDICADOR ">> "
#define CONTINUA ".. "


char *String_copy(char *str) {
	size_t len = strlen(str);
	char *string = gab_alloc(len + 1, 1);
	memmove(string, str, len);
	string[len] = 0;
	return string;
}


int l_closeConsole(lua_State *L) {
	lua_getglobal(L, "CONSOLE_CAN_RUN");
	int *runing = lua_touserdata(L, -1);
	*runing = 0;

	return 0;
}

#define  LUA_CONSOLE_INFO(c) printf("Gabriela cnn versão %s. " \
        LUA_VERSION "\n"                                                      \
        "console commands:\n"\
        "cls      : clear screen\n"\
        "clc      : clear screen\n"\
        "clear    : remove all layers\n"\
        "show     : show cnn\n"\
        "exit     : close console\n"\
        "help     : call helpCnn()\n", c->version)

void enableUtf8() {
	system("chcp 65001|echo off");
	system("echo on");
}

int CnnLuaConsole(Cnn c) {
	if (!c) { return 2; }
	if (!c->LuaVm) { CnnInitLuaVm(c); }
	if (c->ecx->error) { return c->ecx->error; }
	enableUtf8();
	lua_State *L = c->LuaVm;
	int console_run = 1;
	lua_pushlightuserdata(L, &console_run);
	lua_setglobal(L, "CONSOLE_CAN_RUN");
	lua_pushcfunction(L, l_closeConsole);
	lua_setglobal(L, "closeConsole");
//	lua_console("Gab",L);

	Comando cmd = {0};
	int error;
	int ch = 0;
	cmd.str = gab_alloc(1, 0);
	cmd.len = 1;
	LUA_CONSOLE_INFO(c);
	while (console_run) {
		printf(INDICADOR);
		cmd.n = 0;
		cmd.str[cmd.n] = 0;
		while (1) {
			fread(&ch, 1, 1, stdin);

			if (ch == '\n' || ch == '\r') {
				if (GetKeyState(VK_SHIFT) & 0x8000) {
					printf(CONTINUA);
					printf("\n");
				} else {
					printf("\n");
					break;
				}
			}

			if (cmd.n <= cmd.len - 1) {
				cmd.len++;
				cmd.str = gab_realloc(cmd.str, cmd.len);
			}
			cmd.str[cmd.n] = ch;
			cmd.n++;
			cmd.str[cmd.n] = 0;
		}
		if (!cmd.str[0]) { continue; }
		fflush(stdout);
		if (!strcmp(cmd.str, "exit")) { break; }
		if (!strcmp(cmd.str, "cls") || !strcmp(cmd.str, "clc")) {
			system("cls");
			LUA_CONSOLE_INFO(c);
			continue;
		}
		if (!strcmp(cmd.str, "show")) {
			c->print(c, "--");
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
		error = luaL_dostring(c->LuaVm, cmd.str);

		if (error) {
			fflush(stdout);
			fprintf(stderr, "\n Error: %d %d %s\n", lua_gettop(c->LuaVm), error, lua_tostring(c->LuaVm, -1));
			fflush(stderr);

		}
	}

	gab_free(cmd.str);


	lua_pushnil(L);
	lua_setglobal(L, "CONSOLE_CAN_RUN");
	lua_pushnil(L);
	lua_setglobal(L, "closeConsole");
}