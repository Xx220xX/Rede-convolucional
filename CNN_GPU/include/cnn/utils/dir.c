//
// Created by Henrique on 31-Jul-21.
//

#include "dir.h"
#include "windows.h"

void GetCurrentDir(char *buff_dest, size_t size_buff) {
	GetCurrentDirectoryA(size_buff,buff_dest);
}

int SetDir(char *path) {
	return SetCurrentDirectoryA(path);
}
