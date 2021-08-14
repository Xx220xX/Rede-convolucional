//
// Created by Henrique on 31-Jul-21.
//

#include "utils/list_args.h"
void List_argspushValue(List_args *lst,const char *name, const char *value) {
	int i;
	int lenName;
	int lenValue = strlen(value);
	for (i = 0; i < lst->size; ++i) {
		if (strcmp(name, lst->values[i].name) == 0) {
			break;
		}
	}
	if (i < lst->size) {
		free_mem(lst->values[i].value);
		lst->values[i].value = alloc_mem(1, lenValue + 1);
		strcpy(lst->values[i].value, value);
		return;
	}
	lenName = strlen(name);
	lst->size++;
	lst->values = realloc_mem(lst->values, lst->size * sizeof(Dbchar_p));
	lst->values[i].value = alloc_mem(1, lenValue + 1);
	strcpy(lst->values[i].value, value);
	lst->values[i].name = alloc_mem(1, lenName + 1);
	strcpy(lst->values[i].name, name);
}
void releaseList_args(List_args *largs) {
	for (int i = 0; i < largs->size; ++i) {
		free_mem(largs->values[i].name);
		free_mem(largs->values[i].value);
	}
	if (largs->values)
		free_mem(largs->values);
	largs->values = NULL;
	largs->size = 0;
}