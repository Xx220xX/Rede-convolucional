//
// Created by Henrique on 31-Jul-21.
//

#ifndef CNN_GPU_DIR_H
#define CNN_GPU_DIR_H
#include <time.h>
void GetCurrentDir(char *buff_dest,size_t size_buff);
int SetDir(char *path);
int DirectoryExists(const char * szPath) ;
void createDir(const char * dirName);
#endif //CNN_GPU_DIR_H
