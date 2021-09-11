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
	char self_release;
} Dictionary;

void Dict_push(Dictionary *lst, const char *name, const char *value);

void releaseDictionary(Dictionary *largs);


#endif //CNN_GPU_LIST_ARGS_H
