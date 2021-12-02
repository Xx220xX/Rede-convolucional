//
// Created by Henrique on 31-Jul-21.
//

#include <direct.h>
#include "cnn/utils/dir.h"
#include "windows.h"

void GetCurrentDir(char *buff_dest, size_t size_buff) {
	GetCurrentDirectoryA(size_buff, buff_dest);
}

int SetDir(char *path) {
	return !SetCurrentDirectoryA(path);
}

int DirectoryExists(const char * szPath) {
	DWORD dwAttrib = GetFileAttributes(szPath);
	return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
			(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

void createDir(const char *dirName) {
	if(!DirectoryExists(dirName)){
		mkdir(dirName);
	}
}
void resetDir(const char *dirName){
	if(DirectoryExists(dirName)){
		remove(dirName);
	}
	mkdir(dirName);
}