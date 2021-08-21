//
// Created by Henrique on 31-Jul-21.
//

#ifndef CNN_GPU_LIST_ARGS_H
#define CNN_GPU_LIST_ARGS_H

#include <string.h>
#include "memory_utils.h"

#define STREQUALS(x, y)(!strcmp(x,y))
typedef struct {
	char *name;
	char *value;
} Dbchar_p;
typedef struct List_args {
	Dbchar_p *values;
	int size;
} List_args;

void List_argspushValue(List_args *lst,const char *name,const char *value);

void releaseList_args(List_args *largs);


#endif //CNN_GPU_LIST_ARGS_H
